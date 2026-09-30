#!/usr/bin/env bash
# The 12 runs behind docs/REPLAN.md, serial (shared checkpoints.sqlite).
# --margin 0.5 puts the bar past the first attempt.
# --keep-models all: iteration 1 is re-scored later.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# Smallest first, so breakage is cheapest to find.
DATASETS="spambase speeddating adult"

one() {
  local ds=$1 arm=$2 seed=$3
  local extra=""
  [ "$arm" = "nollm" ] && extra="--no-llm"
  local tid="hard${arm}-${ds}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${ds}-seed${seed}.json" \
    --metric balanced_accuracy \
    --goal-mode auto --margin 0.5 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    $extra
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for ds in $DATASETS; do
  for seed in 42 43 44; do
    one "$ds" nollm "$seed"
  done
done

for ds in $DATASETS; do
  one "$ds" llm 42
done

echo "=== 열두 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
