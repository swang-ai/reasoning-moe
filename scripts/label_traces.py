import argparse
import asyncio

from moe_steer.labeling import label_traces
from moe_steer.models import get_model_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-30b-a3b-thinking",
                    help="model key/name that generated the traces (sets the tokenizer)")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--labels", required=True, help="output dir for <id>.json")
    ap.add_argument("--judge", default="gpt-5.4-mini", help="OpenAI-compatible judge model")
    ap.add_argument("--tokenizer", default=None,
                    help="override tokenizer (default: the --model's HF path)")
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--domain-defs", choices=["math", "if", "gpqa", "apps"], default="math",
                    help="phase-definition scheme: math (default) | if | gpqa | apps")
    args = ap.parse_args()

    tokenizer = args.tokenizer
    if tokenizer is None:
        try:
            tokenizer = get_model_spec(args.model).name
        except KeyError:
            tokenizer = args.model

    from moe_steer.labeling import LABEL_DEFINITIONS_BY_DOMAIN
    defs = LABEL_DEFINITIONS_BY_DOMAIN[args.domain_defs]
    print(f"labeling traces in {args.traces}  judge={args.judge}  tokenizer={tokenizer}  "
          f"defs={args.domain_defs}")
    asyncio.run(label_traces(args.traces, args.labels, model=args.judge,
                             max_concurrency=args.max_concurrency,
                             tokenizer_name=tokenizer, label_definitions=defs))


if __name__ == "__main__":
    main()
