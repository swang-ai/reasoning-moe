from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

MOE_BLOCK_CLASSES = ("Qwen3NextSparseMoeBlock", "Qwen3MoeSparseMoeBlock",
                     "DeepseekV2MoE", "DeepseekV3MoE",
                     "MLPBlock")
ROUTER_CLASSES = ("Gemma4Router",)


def _worker_install(self, top_k=None):
    model = self.model_runner.model
    if top_k is None:
        cfg = getattr(getattr(model, "language_model", model), "config", None)
        top_k = int(getattr(cfg, "num_experts_per_tok", 8)) if cfg is not None else 8
    self._trc_topk = int(top_k)
    blocks = []
    for name, m in model.named_modules():
        cn = m.__class__.__name__
        g = None
        if cn in MOE_BLOCK_CLASSES and (getattr(m, "gate", None) is not None or getattr(m, "router", None) is not None):
            g = getattr(m, "gate", None) or m.router
        elif cn in ROUTER_CLASSES:
            g = m
        if g is None:
            continue
        mnum = re.search(r"layers\.(\d+)\.", name)
        blocks.append((int(mnum.group(1)) if mnum else len(blocks), name, g))
    blocks.sort(key=lambda b: b[0])
    self._trc_buffers = [[] for _ in blocks]
    self._trc_handles = []
    for li, (_, _, g) in enumerate(blocks):
        def mk(idx):
            def hook(module, inp, out):
                rl = out[0] if isinstance(out, tuple) else out
                ids = torch.topk(rl.float(), k=self._trc_topk, dim=-1).indices
                self._trc_buffers[idx].append(ids.detach().to(torch.int16).cpu())
            return hook
        self._trc_handles.append(g.register_forward_hook(mk(li)))
    return len(blocks)


def _worker_reset(self):
    for b in self._trc_buffers:
        b.clear()
    return True


def _worker_collect_slice(self, start, count):
    layers = []
    for buf in self._trc_buffers:
        if not buf:
            return None
        full = torch.cat(buf, dim=0)
        sl = full[start:start + count, :]
        if sl.shape[0] != count:
            return None
        layers.append(sl)
    return torch.stack(layers, dim=0).numpy()


FORCE_WRAP = ("\n\nConsidering the limited time, let me stop reasoning and give the "
              "final answer based on what I have.\n</think>\n\n")


def extract_traces_vllm(llm, prompts, out_traces_dir, max_new_tokens=8192,
                        think_budget=None, answer_budget=2048, force_wrap=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    out_traces_dir = Path(out_traces_dir); out_traces_dir.mkdir(parents=True, exist_ok=True)
    tok = llm.get_tokenizer()
    force = force_wrap if force_wrap is not None else FORCE_WRAP

    todo = []
    for i, p in enumerate(prompts):
        idx = int(p.get("_id", i))
        if not (out_traces_dir / f"{idx:05d}.pt").exists():
            todo.append((idx, p))
    if not todo:
        print("[vllm-trace] all traces already present", flush=True)
        return 0

    chats = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}],
                                     add_generation_prompt=True, tokenize=False)
             for _, p in todo]
    cap = think_budget or max_new_tokens
    print(f"[vllm-trace] phase1: batch-generating {len(todo)} CoTs (cap={cap}"
          f"{', budget-forcing' if think_budget else ''})...", flush=True)
    outs = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=cap), use_tqdm=True)

    finals, forced = [], []
    for (idx, p), out in zip(todo, outs):
        pids = list(out.prompt_token_ids); gids = list(out.outputs[0].token_ids)
        txt = out.outputs[0].text
        if think_budget and "</think>" not in txt:
            forced.append((idx, p, pids, gids, txt))
        else:
            finals.append((idx, p, pids, gids, txt))

    if forced:
        force_ids = tok(force, add_special_tokens=False).input_ids
        print(f"[vllm-trace] budget-forcing {len(forced)}/{len(todo)} truncated rollouts "
              f"-> wrap-up + answer pass...", flush=True)
        prompts2 = [TokensPrompt(prompt_token_ids=pids + gids + force_ids)
                    for _, _, pids, gids, _ in forced]
        outs2 = llm.generate(prompts2, SamplingParams(temperature=0.0, max_tokens=answer_budget),
                             use_tqdm=True)
        for (idx, p, pids, gids, txt), o2 in zip(forced, outs2):
            ans_ids = list(o2.outputs[0].token_ids)
            finals.append((idx, p, pids, gids + force_ids + ans_ids, txt + force + o2.outputs[0].text))

    _rpc = getattr(llm, "collective_rpc", None)
    if _rpc is None:
        _rpc = llm.llm_engine.collective_rpc
    L = _rpc(_worker_install)[0]
    print(f"[vllm-trace] phase2: hooked {L} MoE blocks; capturing routing...", flush=True)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    n_done = 0
    for idx, p, prompt_ids, gen_ids, text in finals:
        n_gen = len(gen_ids)
        if n_gen == 0:
            print(f"[vllm-trace] WARN idx{idx}: empty generation; skip", flush=True)
            continue
        _rpc(_worker_reset)
        llm.generate(TokensPrompt(prompt_token_ids=prompt_ids + gen_ids), sp1, use_tqdm=False)
        ids = _rpc(_worker_collect_slice,
                                            args=(len(prompt_ids), n_gen))[0]
        if ids is None:
            print(f"[vllm-trace] WARN idx{idx}: bad capture (n_gen={n_gen}); skip", flush=True)
            continue
        torch.save({"topk_ids": torch.from_numpy(ids),
                    "gen_token_ids": torch.tensor(gen_ids),
                    "problem": p["problem"], "gold": p.get("gold"),
                    "text": text}, out_traces_dir / f"{idx:05d}.pt")
        n_done += 1
        if n_done % 20 == 0:
            print(f"[vllm-trace] {n_done}/{len(finals)} routed", flush=True)
    print(f"[vllm-trace] DONE: {n_done} new traces -> {out_traces_dir}", flush=True)
    return n_done
