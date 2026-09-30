#!/usr/bin/env bash
# The 12 runs behind docs/WEIGHT-START.md; rules fallback only.
# Same flags as run_hard_bar.sh's free arm; CPU only.
# Recorded bar65llm and bar65nollm are reused, not re-run.
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in bench/hard_bar.py DATASETS order; binary only.
SETS="adult:balanced_accuracy \
bank-marketing:balanced_accuracy \
speeddating:balanced_accuracy \
spambase:balanced_accuracy"

one() {
  local dataset=$1 metric=$2 seed=$3
  local tid="bar65nollmratio-${dataset}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${dataset}-seed${seed}.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.65 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    --no-llm
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for seed in 42 43 44; do
  for pair in $SETS; do
    one "${pair%%:*}" "${pair##*:}" "$seed"
  done
done

echo "=== 열두 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
echo "=== 판정: python -m bench.weight_start ==="
