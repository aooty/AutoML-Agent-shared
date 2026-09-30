#!/usr/bin/env bash
# The 25 runs behind docs/REPEATS.md, serial (shared checkpoints.sqlite).
# R arms differ only in --thread-id; S arms change seed.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

one() {
  local ds=$1 metric=$2 rep=$3 seed=$4
  local tid="rep${rep}-${ds}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${ds}-seed${seed}.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.25 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models best \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid"
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

# Smallest first, so breakage is cheapest to find.
DATASETS=("spambase balanced_accuracy" "speeddating balanced_accuracy" \
          "house_sales r2" "bank-marketing balanced_accuracy" "adult balanced_accuracy")

# R: three repeats of one setting, all before S.
for rep in 1 2 3; do
  for pair in "${DATASETS[@]}"; do
    set -- $pair
    one "$1" "$2" "$rep" 42
  done
done

# S: the seed lever, repeat 1 only.
for seed in 43 44; do
  for pair in "${DATASETS[@]}"; do
    set -- $pair
    one "$1" "$2" 1 "$seed"
  done
done

echo "=== 스물다섯 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
