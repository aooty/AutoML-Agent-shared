#!/usr/bin/env bash
# The 24 runs behind docs/HARD-BAR.md, serial (shared checkpoints.sqlite).
# Same flags as run_prompt_redo.sh, but --margin 0.65.
# Free arm runs first; check bars before the paid half.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in bench/hard_bar.py DATASETS order; binary only.
SETS="adult:balanced_accuracy \
bank-marketing:balanced_accuracy \
speeddating:balanced_accuracy \
spambase:balanced_accuracy"

one() {
  local arm=$1 dataset=$2 metric=$3 seed=$4
  local extra=""
  [ "$arm" = "bar65nollm" ] && extra="--no-llm"
  local tid="${arm}-${dataset}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${dataset}-seed${seed}.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.65 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    $extra
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for seed in 42 43 44; do
  for pair in $SETS; do
    one bar65nollm "${pair%%:*}" "${pair##*:}" "$seed"
  done
done

echo "=== 무료 열두 번 끝났습니다 ($(date '+%H:%M:%S')) ==="
echo "=== 유료 절반 전에 바를 대조하십시오: python -m bench.hard_bar ==="

for seed in 42 43 44; do
  for pair in $SETS; do
    one bar65llm "${pair%%:*}" "${pair##*:}" "$seed"
  done
done

echo "=== 스물네 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
