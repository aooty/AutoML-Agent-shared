#!/usr/bin/env bash
# The twelve runs behind docs/REPLAN.md: 3 datasets x (hard_nollm seeds 42/43/44 + hard_llm
# seed 42). Sequential on purpose: the arms share one checkpoints.sqlite, and two writers race
# for its lock.
#
# --keep-models all is not a convenience here. The pre-registered criterion re-scores each run's
# iteration 1 on the test rows, and the default (best) deletes every non-winning iteration's
# model.joblib — bench/replan.py refuses the dataset rather than silently dropping the pair.
#
# --margin 0.5 puts the bar out of reach of the first attempt on all three (measured on the
# committed seed-42 cards: adult 0.8832, spambase 0.9614, speeddating 0.8342 against recorded
# iteration-1 val 0.8481 / 0.9484 / 0.7187). That the bar was raised deliberately is the
# disclosed condition of this experiment, not a footnote.
#
# The free arm runs first, all nine of it. If anything about this pipeline is broken, it costs
# CPU to find out instead of Opus tokens.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across thread
# counts, 0.99x the half-width of one paired verdict. Bedrock route because that is what every
# recorded llm arm used (route=bedrock, model=anthropic.claude-opus-5).
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# Smallest first, so a broken pipeline costs the least to discover.
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
