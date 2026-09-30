#!/usr/bin/env bash
# The ten runs behind docs/SPG.md. Sequential on purpose: the arms share one
# checkpoints.sqlite, and two writers race for its lock.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across
# thread counts, 0.99x the half-width of one paired verdict. Bedrock route because that is
# what the recorded arm used (route=bedrock, model=anthropic.claude-opus-5).
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

one() {
  local ds=$1 arm=$2 metric=$3
  local extra=""
  [ "$arm" = "on" ] && extra="--search-past-goal"
  local tid="spg${arm}-${ds}-seed42"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${ds}-seed42.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.25 \
    --max-iterations 5 --time-budget-sec 3600 --seed 42 \
    --keep-models best \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    $extra
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

# Smallest first, so a broken pipeline costs the least to discover.
for pair in "spambase balanced_accuracy" "speeddating balanced_accuracy" \
            "house_sales r2" "bank-marketing balanced_accuracy" "adult balanced_accuracy"; do
  set -- $pair
  one "$1" off "$2"
  one "$1" on "$2"
done

echo "=== 열 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
