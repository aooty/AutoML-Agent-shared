#!/usr/bin/env bash
# The 12 runs behind docs/PROMPT-REDO.md, serial (shared checkpoints.sqlite).
# Recorded arms' settings, re-run under today's prompt.
# Free arms run first, so breakage costs CPU, not tokens.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in bench/prompt_redo.py DATASETS order.
SETS="adult:balanced_accuracy \
bank-marketing:balanced_accuracy \
speeddating:balanced_accuracy \
spambase:balanced_accuracy \
house_sales:r2 \
jungle-chess:balanced_accuracy"

one() {
  local arm=$1 dataset=$2 metric=$3
  local extra=""
  [ "$arm" = "nollm2" ] && extra="--no-llm"
  local tid="${arm}-${dataset}-seed42"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${dataset}-seed42.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.25 \
    --max-iterations 5 --time-budget-sec 3600 --seed 42 \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    $extra
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for pair in $SETS; do
  one nollm2 "${pair%%:*}" "${pair##*:}"
done

for pair in $SETS; do
  one llm2 "${pair%%:*}" "${pair##*:}"
done

echo "=== 열두 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
