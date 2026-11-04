import argparse
import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--domain", default="mathmix_find")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--think-budget", type=int, default=None,
                    help="s1-style budget forcing: cap CoT here; truncated rollouts get a "
                         "forced wrap-up + answer pass so every trace reaches a conclusion.")
    ap.add_argument("--answer-budget", type=int, default=2048,
                    help="max tokens for the forced answer pass (only with --think-budget)")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    from moe_steer.datasets import get_domain
    from moe_steer.traces_vllm import extract_traces_vllm
    from vllm import LLM

    prompts = get_domain(args.domain).load(n=args.n, seed=args.seed)
    for gid, p in enumerate(prompts):
        p["_id"] = gid
    prompts = prompts[args.shard::args.shards]
    print(f"shard {args.shard}/{args.shards}: {len(prompts)} prompts", flush=True)

    ctx = (args.think_budget + args.answer_budget) if args.think_budget else args.max_new_tokens
    mml = args.max_model_len or (ctx + 2048)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem, max_model_len=mml,
              enforce_eager=True, enable_prefix_caching=False,
              trust_remote_code=True, max_num_seqs=64)
    extract_traces_vllm(llm, prompts, args.out, max_new_tokens=args.max_new_tokens,
                        think_budget=args.think_budget, answer_budget=args.answer_budget)


if __name__ == "__main__":
    main()
