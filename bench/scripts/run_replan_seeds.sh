#!/usr/bin/env bash
# The 6 runs behind docs/REPLAN-SEEDS.md, serial (shared checkpoints.sqlite).
# Speeddating at seeds 42/43/44, same condition as REPLAN.md.
# Suffix 2 keeps recorded REPLAN.md archives untouched.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

DATASET=speeddating

one() {
  local arm=$1 seed=$2
  local extra=""
  [ "$arm" = "nollm2" ] && extra="--no-llm"
  local tid="hard${arm}-${DATASET}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${DATASET}-seed${seed}.json" \
    --metric balanced_accuracy \
    --goal-mode auto --margin 0.5 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    $extra
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for seed in 42 43 44; do
  one nollm2 "$seed"
done

for seed in 42 43 44; do
  one llm2 "$seed"
done

echo "=== 여섯 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
