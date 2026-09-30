#!/usr/bin/env bash
# The 25 runs behind docs/REPEATS.md: 15 repeats at the judged seed (R) and 10 at seeds
# 43/44 (S). Sequential on purpose: the arms share one checkpoints.sqlite, and two writers
# race for its lock.
#
# The R arms differ in nothing but --thread-id, which is the checkpoint key and reaches no
# prompt. That is the point: their planning prompts are byte-identical, so what the deltas
# carry is the model answering the same question twice.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across
# thread counts, 0.99x the half-width of one paired verdict — the same size as what this
# experiment is trying to measure. Bedrock route because that is what every recorded arm used.
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

# Smallest first, so a broken pipeline costs the least to discover.
DATASETS=("spambase balanced_accuracy" "speeddating balanced_accuracy" \
          "house_sales r2" "bank-marketing balanced_accuracy" "adult balanced_accuracy")

# R — three repeats of one configuration. All three before S, so that a run interrupted
# halfway still answers the question that gates reading everything else.
for rep in 1 2 3; do
  for pair in "${DATASETS[@]}"; do
    set -- $pair
    one "$1" "$2" "$rep" 42
  done
done

# S — the seed lever. Repeat 1 only; the cards for these seeds are already committed.
for seed in 43 44; do
  for pair in "${DATASETS[@]}"; do
    set -- $pair
    one "$1" "$2" 1 "$seed"
  done
done

echo "=== 스물다섯 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
