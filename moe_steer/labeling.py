from __future__ import annotations

import asyncio
import json
import os
import re

_JUDGE_KW = json.loads(os.environ.get("LABEL_JUDGE_KW", "{}"))
from dataclasses import dataclass
from pathlib import Path

from .data_utils import PHASE_LABELS

VALID_LABELS = set(PHASE_LABELS)

LABEL_DEFINITIONS = """\
- planning:    Meta-statements about what to do next. Signals like "Let's...",
               "First, we need to...", "Now let's try...", "So the plan is...".
               Content is about the *approach*, not the math itself.
- reasoning:   Actual derivation steps — applying a rule, doing algebra,
               computing a value, drawing a logical inference. Includes LaTeX
               equations that are part of the derivation chain. NOT an opinion
               about a step (that is reflection).
- restatement: Repeating or paraphrasing the problem statement or a previously
               established fact from earlier in the trace.
- fact:        Invoking external knowledge not given in the problem — a named
               theorem, a general formula, a definition.
- reflection:  DOUBT or dissatisfaction about prior steps — uncertainty,
               confusion, hesitation, or a suspicion that something is wrong —
               that triggers reconsideration and is REMOVABLE without breaking the
               derivation. e.g. "But I'm not sure about this.", "Hmm, wait.",
               "This seems off.", "That doesn't look right.", "I'm stuck." If it
               states what to do next ("Let's try X") -> planning; if it
               computes/derives -> reasoning; if it expresses SATISFACTION,
               confirms a result is correct, or signals wrapping up -> conclusion.
- conclusion:  The final answer, an intermediate result, OR a closure/commitment
               signal that the model is satisfied and DONE reasoning — the
               positive "off-ramp". e.g. "Stop. Enough.", "This confirms the
               earlier result.", "That's correct.", "Looks right.", "Good, that
               works.", "So the answer is ...", "Done." (Contrast reflection,
               which is the NEGATIVE opinion that keeps reasoning going.)
"""

LABEL_DEFINITIONS_IF = """\
- planning:    Meta-statements about how to approach the WRITING task — "First
               I'll draft...", "I'll structure it as...", "Let me lay out the
               format...", "The plan is to...". About HOW to compose/organize, not
               the content itself.
- reasoning:   The actual composition WORK — drafting or translating the response,
               deciding wording/phrasing/content, generating the text that answers
               the request. NEW work; not checking a constraint (reflection) and
               not repeating the request (restatement).
- restatement: Repeating or paraphrasing the request or, especially, its
               CONSTRAINTS / requirements — "They want exactly two responses, no
               commas, in Bulgarian", "The instruction says 300+ words".
- fact:        Invoking external knowledge NOT given in the prompt to do the task
               — a recalled world fact, a translation, a name/date, a definition.
- reflection:  DOUBT or self-checking about whether the output meets the
               CONSTRAINTS — "wait, did I use a comma?", "is this exactly 300
               words?", "do I have three highlighted sections?", "hmm, that might
               not be allowed", uncertainty about satisfying a requirement, that
               triggers a fix. If it says what to do next -> planning; if it
               drafts content -> reasoning; if it confirms all constraints are met
               / done -> conclusion.
- conclusion:  The final response/output, OR a positive closure signal that every
               requirement is satisfied and the model is DONE — "that satisfies
               all the constraints", "this is exactly two responses", "Done.",
               "so here is the answer". (Contrast reflection, the NEGATIVE doubt
               that keeps it revising.)
"""

LABEL_DEFINITIONS_GPQA = """\
- planning:    Meta-statements about how to approach the science problem — "Let's
               identify what's being asked", "I'll use conservation of energy",
               "First find the relevant principle/equation", "The approach is...".
- reasoning:   The actual scientific derivation/work — applying a physical law,
               doing the algebra/arithmetic, propagating units, deriving a
               quantity, or ruling out an option on mechanistic grounds. NEW work.
- restatement: Repeating or paraphrasing the question, the answer options, or the
               given quantities/conditions.
- fact:        Invoking external scientific knowledge NOT given in the problem — a
               named law, constant, formula, reaction, or definition (e.g. "the
               Rydberg constant is...", "by the Pauli exclusion principle...").
- reflection:  DOUBT or self-checking about prior steps — "wait, is that the right
               formula?", "let me double-check the units", "hmm, that magnitude
               seems off", reconsidering which option fits, uncertainty about a
               calculation that triggers rechecking. If it says what to do next ->
               planning; if it derives -> reasoning; if it confirms/commits ->
               conclusion.
- conclusion:  The final choice/answer or a positive closure — "so the answer is
               C", "this matches option B", "that's consistent, done". (Contrast
               reflection, the negative doubt that keeps it rechecking.)
"""

LABEL_DEFINITIONS_APPS = """\
- planning:    Algorithm/approach design — "I'll use a hash map", "this is a DP
               problem", "let me think about complexity", "plan: sort then
               two-pointer", "I need to handle these edge cases". The STRATEGY,
               not the code itself.
- reasoning:   The actual problem-solving WORK — working through the algorithm,
               tracing an example, deriving the recurrence, or writing/deciding
               the code logic that implements the solution. NEW work.
- restatement: Repeating or paraphrasing the problem statement, the constraints,
               or the input/output format.
- fact:        Invoking known CS knowledge NOT given — a standard algorithm or
               data structure and its properties, a complexity bound, a library
               function's behavior ("Dijkstra is O(E log V)", "a set gives O(1)
               lookup", "sorting is O(n log n)").
- reflection:  DEBUGGING doubt / self-checking — "wait, this fails on empty
               input", "does this handle negatives?", "off-by-one here?", "let me
               re-check the edge cases", "hmm, this might TLE", uncertainty that
               triggers a fix. If it says what to do next -> planning; if it works
               through logic -> reasoning; if it confirms correctness -> conclusion.
- conclusion:  The final solution/code or a positive closure — "this should
               work", "that handles all cases", "so here's the final code".
               (Contrast reflection, the doubt that keeps it debugging.)
"""

LABEL_DEFINITIONS_BY_DOMAIN = {
    "math": LABEL_DEFINITIONS,
    "if":   LABEL_DEFINITIONS_IF,
    "gpqa": LABEL_DEFINITIONS_GPQA,
    "apps": LABEL_DEFINITIONS_APPS,
}

SEG_MARKER = "<<|SEG|>>"

SEGMENTATION_PROMPT = f"""\
TASK: Copy the reasoning trace below VERBATIM, with the marker `{SEG_MARKER}`
inserted between adjacent semantic spans. That is the ONLY thing you may add.

================================================================
CRITICAL — VIOLATING ANY OF THESE WILL FAIL THE TASK:
================================================================
1. Do NOT change ANY character. Not whitespace, not punctuation, not words.
2. Do NOT "fix" typos. Do NOT normalize punctuation (e.g. "->" stays "->",
   never "=>"; "But maybe" stays "But maybe", never "However").
3. Do NOT add or drop ANY characters — including trailing tokens like
   "assistantfinal", trailing whitespace, trailing newlines.
4. Do NOT include the `<<<TRACE` / `TRACE>>>` delimiters in your output.
5. Do NOT change spaces to newlines or newlines to spaces.
6. The ONLY characters you may insert are the marker `{SEG_MARKER}`.
7. Do NOT rewrite, paraphrase, continue, complete, or summarize the reasoning —
   even if it looks unfinished, wrong, or could be phrased better. Output the
   trace's exact words, nothing of your own.

VERIFICATION: when the markers are removed from your output, the result MUST
be byte-identical to the input trace. We check this and reject mismatches.
================================================================

Each span should express ONE of:
  - a planning statement
  - a reasoning step (one derivation, one equation, one inference)
  - a restatement of the problem or an earlier fact
  - an external fact (named theorem, general formula)
  - a conclusion

Spans:
  - are typically one clause to one sentence
  - are non-overlapping and cover the entire trace
  - LaTeX display equations (\\[ ... \\]) are usually their own span
  - do NOT split a single equation across spans

================================================================
EXAMPLES (study these — these are the mistakes you must avoid):
================================================================

INPUT:
analysisLet me solve x + 5 = 10. So x -> 10 - 5 = 5.

Thus x = 5.assistantfinalThe answer is 5.

CORRECT OUTPUT (only markers added, every character preserved):
analysis{SEG_MARKER}Let me solve x + 5 = 10. {SEG_MARKER}So x -> 10 - 5 = 5.

{SEG_MARKER}Thus x = 5.{SEG_MARKER}assistantfinal{SEG_MARKER}The answer is 5.

WRONG examples — DO NOT do these:
  ❌ "So x → 10 - 5 = 5"          (changed "->" to "→")
  ❌ "However x = 5"               (replaced "Thus" with "However")
  ❌ ". {SEG_MARKER}\nSo x"        (added newline that wasn't there)
  ❌ ".{SEG_MARKER}So x"           (dropped the space after period)
  ❌ "The answer is 5."            (dropped "assistantfinal" prefix)
  ❌ trailing "TRACE>>>"            (kept the input delimiter)
  ❌ "x = 5. Thus x = 5."          (deduplicated/normalized text)
  ❌ "No.[newline]So final answer" (turned a space into a newline — keep spaces)
  ❌ rewriting "Let's provide reasoning" as "But could mean something else:"
     (INVENTED text — copy verbatim, never continue or paraphrase)

================================================================

Trace to segment (DO NOT MODIFY):
<<<TRACE
{{trace}}
TRACE>>>

Output the trace VERBATIM with `{SEG_MARKER}` inserted between spans. No
preamble, no code fences, no commentary, no closing delimiter.
"""

CLASSIFICATION_PROMPT = """\
Label atomic spans from a reasoning trace. Assign each span exactly ONE label.

{label_definitions}

Full trace for context:
<<<TRACE
{trace}
TRACE>>>

Spans to label:
{numbered_spans}

Output format — exactly one line per span, '<n>. <label>':
1. <label>
2. <label>
...

Valid labels: {allowed}.
Output ALL {n} labels in order, one per line, no extra text.
"""


@dataclass
class Span:
    start: int
    end: int
    text: str
    is_code: bool = False


def token_char_offsets(tok, token_ids, window=32):
    ids = [int(t) for t in token_ids]
    text = tok.decode(ids, skip_special_tokens=True)
    if getattr(tok, "is_fast", False):
        try:
            special = set(tok.all_special_ids)
            nonsp = [t for t in ids if t not in special]
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            om = enc["offset_mapping"]
            if len(om) == len(nonsp):
                offsets, j, last = [], 0, 0
                for t in ids:
                    if t in special:
                        offsets.append((last, last))
                    else:
                        s, e = om[j]; offsets.append((int(s), int(e)))
                        last = int(e); j += 1
                return offsets, text
        except Exception:
            pass
    offsets, pos = [], 0
    for i in range(len(ids)):
        lo = max(0, i + 1 - window)
        cur = len(tok.decode(ids[lo:i + 1], skip_special_tokens=True))
        prevw = len(tok.decode(ids[lo:i], skip_special_tokens=True)) if i > lo else 0
        delta = cur - prevw
        offsets.append((pos, pos + delta))
        pos += delta
    return offsets, text


def char_span_to_token_range(offsets, c_start, c_end):
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if e > c_start and tok_start is None:
            tok_start = i
        if s < c_end:
            tok_end = i + 1
    if tok_start is None or tok_end is None or tok_start >= tok_end:
        return None
    return tok_start, tok_end


def _sentence_spans(region, region_start, max_seg_chars=None):
    stripped = region.strip()
    if not stripped:
        return []
    lead = region.find(stripped)
    parts, prev = [], 0
    for mo in re.finditer(r"(?<=[.!?\n])\s+", stripped):
        if mo.start() > prev:
            parts.append((prev, mo.start()))
        prev = mo.end()
    if len(stripped) > prev:
        parts.append((prev, len(stripped)))

    def _span(a, b):
        la, lb = a + lead, b + lead
        return Span(region_start + la, region_start + lb, region[la:lb])

    if not max_seg_chars:
        return [_span(s, e) for s, e in parts]

    spans, cur_len, cs, ce = [], 0, None, None
    for s, e in parts:
        plen = e - s
        if cs is not None and (cur_len + plen) > max_seg_chars:
            spans.append(_span(cs, ce)); cur_len, cs = 0, None
        if cs is None:
            cs = s
        cur_len += (1 if cur_len else 0) + plen
        ce = e
    if cs is not None:
        spans.append(_span(cs, ce))
    return spans


def segment_spans(text, max_seg_chars=None):
    spans, pos = [], 0
    for m in re.finditer(r"```.*?```", text, re.S):
        if m.start() > pos:
            spans.extend(_sentence_spans(text[pos:m.start()], pos, max_seg_chars))
        spans.append(Span(m.start(), m.end(), text[m.start():m.end()], is_code=True))
        pos = m.end()
    if pos < len(text):
        spans.extend(_sentence_spans(text[pos:], pos, max_seg_chars))
    return spans


async def segment_trace(client, trace, model, **kw):
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SEGMENTATION_PROMPT.format(trace=trace)}],
        **kw,
    )
    raw = resp.choices[0].message.content
    raw = re.sub(r"^```\w*\s*\n?|\n?```\s*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\s*<<<TRACE\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*TRACE>>>\s*$", "", raw)
    raw = re.sub(r"\n?\s*TRACE\s*$", "", raw)

    pieces = [p for p in raw.split(SEG_MARKER) if p]
    reconstructed = "".join(pieces)
    if reconstructed != trace and reconstructed.rstrip() != trace.rstrip():
        diff_at = next((i for i, (a, b) in enumerate(zip(reconstructed, trace)) if a != b),
                       min(len(reconstructed), len(trace)))
        raise ValueError(
            f"Reconstruction mismatch at char {diff_at}: "
            f"got {reconstructed[max(0,diff_at-30):diff_at+30]!r} "
            f"vs {trace[max(0,diff_at-30):diff_at+30]!r}")

    spans, cursor = [], 0
    for piece in pieces:
        end = cursor + len(piece)
        spans.append(Span(cursor, end, piece))
        cursor = end
    if not spans:
        raise ValueError("Segmentation produced 0 spans")
    return spans


async def classify_spans(client, trace, spans, model, batch_size=30,
                         label_definitions=None, **kw):
    label_definitions = label_definitions or LABEL_DEFINITIONS
    all_labels = []
    for bstart in range(0, len(spans), batch_size):
        batch = spans[bstart: bstart + batch_size]
        numbered = "\n".join(f"{i+1}. {s.text!r}" for i, s in enumerate(batch))
        prompt = CLASSIFICATION_PROMPT.format(
            label_definitions=label_definitions, trace=trace,
            numbered_spans=numbered, n=len(batch),
            allowed=", ".join(PHASE_LABELS))
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **{**_JUDGE_KW, **kw})
        all_labels.extend(_parse_labels(resp.choices[0].message.content.strip(), len(batch)))
    assert len(all_labels) == len(spans)
    return all_labels


def _parse_labels(raw, expected):
    labels = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        tok = (m.group(1) if m else line).strip().lower()
        w = re.match(r"[a-z]+", tok)
        if not w:
            continue
        lbl = w.group(0)
        if lbl not in VALID_LABELS:
            raise ValueError(f"Invalid label {lbl!r} in:\n{raw[:300]}")
        labels.append(lbl)
    if len(labels) != expected:
        raise ValueError(f"Expected {expected} labels, parsed {len(labels)}. Raw:\n{raw[:300]}")
    return labels


async def label_one_trace(client, tok, pt_path, model, semaphore, label_definitions=None):
    import torch
    rec = torch.load(pt_path, weights_only=False, map_location="cpu")
    trace_id = rec.get("trace_id", Path(pt_path).stem)
    question = rec.get("problem")
    gen_token_ids = rec["gen_token_ids"]
    response = tok.decode(gen_token_ids, skip_special_tokens=True)
    token_offsets, _ = token_char_offsets(tok, gen_token_ids)

    spans = segment_spans(response)
    async with semaphore:
        labels = await classify_spans(client, response, spans, model,
                                      label_definitions=label_definitions)

    nodes, dropped = [], 0
    for i, (span, label) in enumerate(zip(spans, labels)):
        tr = char_span_to_token_range(token_offsets, span.start, span.end)
        if tr is None:
            dropped += 1
            continue
        t0, t1 = tr
        nodes.append({"id": f"{trace_id}_seg{i:03d}", "label": label,
                      "char_start": span.start, "char_end": span.end,
                      "token_start": t0, "token_end": t1,
                      "text": response[span.start:span.end]})
    return {"trace_id": trace_id, "doc_id": trace_id,
            "raw_text": {"question": question, "response": response},
            "metadata": {"labeler": f"openai:{model}", "n_spans_kept": len(nodes),
                         "n_dropped_token_align": dropped},
            "nodes": nodes}


def save_labels(label_dict, labels_dir, trace_id):
    out = Path(labels_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{trace_id}.json").write_text(json.dumps(label_dict, indent=2))


async def label_traces(traces_dir, out_dir, model="gpt-5.4-mini", max_concurrency=8,
                       tokenizer_name="openai/gpt-oss-20b", label_definitions=None):
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    sem = asyncio.Semaphore(max_concurrency)
    pts = sorted(p for p in Path(traces_dir).glob("*.pt"))
    if not pts:
        raise SystemExit(f"no *.pt traces in {traces_dir}")

    async def process(pt):
        op = out / f"{pt.stem}.json"
        if op.exists():
            return
        try:
            op.write_text(json.dumps(await label_one_trace(client, tok, pt, model, sem,
                                                           label_definitions=label_definitions),
                                     ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"❌ {pt.name}: {type(e).__name__}: {e}")

    await asyncio.gather(*[process(p) for p in pts])
    print(f"wrote labels to {out}/  ({len(list(out.glob('*.json')))}/{len(pts)})")
