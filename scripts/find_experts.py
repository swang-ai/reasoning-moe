import argparse

from moe_steer import load_labeled_traces, write_class_pkls, get_model_spec
from moe_steer.config import EXPERTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", default=str(EXPERTS_DIR))
    ap.add_argument("--metric", choices=["risk_diff"], default="risk_diff",
                    help="risk_diff (difference of hard top-k rates)")
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--num-experts", type=int)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--min-tokens", type=int, default=1000,
                    help="drop phase classes with fewer than this many tokens")
    args = ap.parse_args()

    try:
        s = get_model_spec(args.model)
        L, E, K = args.num_layers or s.num_layers, args.num_experts or s.num_experts, args.top_k or s.top_k
    except KeyError:
        L, E, K = args.num_layers, args.num_experts, args.top_k
        assert L and E and K, "unknown model: pass --num-layers/--num-experts/--top-k"

    router, y, labels_used = load_labeled_traces(args.traces, args.labels, L, E,
                                                 min_tokens=args.min_tokens)
    print(f"model={args.model}  geometry={L}x{E} top-{K}")
    print("\n== risk_diff ==")
    write_class_pkls(args.out_dir, labels_used, router=router, y=y, top_k=K)
    print(f"\nwrote pickles to {args.out_dir}/")


if __name__ == "__main__":
    main()
