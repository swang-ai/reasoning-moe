#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)" STEER_EPS="${STEER_EPS:-0.01}" VLLM_ALLOW_INSECURE_SERIALIZATION=1
MODEL="${MODEL:?eval registry key, e.g. qwen30-mathmix-6c}"
SPEC="${SPEC:?moe_steer registry key, e.g. qwen3-30b-a3b}"
HF="${HF:?HF model id or local path}"
GPUS="${GPUS:-0}"
DOMAIN="${DOMAIN:-mathmix_find}"
TAG="${TAG:-$MODEL}"
TR=data/traces/$TAG; LB=data/labels/$TAG; EX=data/experts/$MODEL-nofilter
TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)

[ "$(ls $TR/*.pt 2>/dev/null | wc -l)" -ge 300 ] || CUDA_VISIBLE_DEVICES=$GPUS \
  python scripts/extract_traces_vllm.py --model-path "$HF" --domain "$DOMAIN" \
    --out $TR --n 300 --max-new-tokens 16384 --tp $TP --gpu-mem 0.90
[ "$(ls $LB/*.json 2>/dev/null | wc -l)" -ge 300 ] || \
  python scripts/label_traces.py --model "$SPEC" --traces $TR --labels $LB \
    --tokenizer "$HF" --domain-defs math --judge "${JUDGE_MODEL:?set JUDGE_MODEL}" --max-concurrency 60
[ -f $EX/riskdiff_reflection.pkl ] || \
  python scripts/find_experts_incremental.py --model "$SPEC" --traces $TR --labels $LB \
    --out-dir $EX --min-tokens 1000 --device cpu
