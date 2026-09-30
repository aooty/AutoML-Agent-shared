#!/usr/bin/env bash
# The twelve runs behind docs/PROMPT-REDO.md: six datasets x (no_llm2 + llm2) at seed 42.
# Sequential on purpose: the arms share one checkpoints.sqlite, and two writers race for its lock.
#
# Why at all: every row of docs/RESULTS.md's headline table was measured under a prompt this
# repository no longer has. d3e7f83 widened MODEL_REGISTRY's params/notes to the executor's actual
# width and 0ef38f8 re-edited the same notes, and that list is rendered verbatim into the planning
# and model-selection prompts. REGISTRY-GAP.md wrote the consequence down itself: "이후 실행을 기존
# 표에 같은 열로 적으면 안 되고, 비교하려면 이 커밋 이후로 여섯 팔을 다시 돌리는 것이
# 필요합니다." This is that.
#
# The settings are the recorded arms' settings, read out of their own run_config.json rather than
# remembered: --goal-mode auto, --margin 0.25 (the default; written out so the condition is on
# screen), --max-iterations 5, --time-budget-sec 3600, --seed 42, Bedrock, threads pinned to 1.
# Changing any of them would make this a new experiment instead of a re-measurement.
#
# The flags are the same; the bar two of them derive is not. 12fa74b floors the auto bar at the
# ranking ceiling, which raised bank-marketing 0.7465 -> 0.8400 and speeddating 0.7514 -> 0.7712,
# so on those two cards these runs are allowed to stop later than the recorded arms were. Nothing
# here forces the old bar back: making a run chase a threshold today's code would not derive is
# constructing a comparison rather than taking one. bench/prompt_redo.py publishes goal_threshold
# per arm and marks the pairs that span the change (spans_goal_change).
#
# On bank-marketing --margin 0.25 is now inert: its floored_margin is 0.526, and 12fa74b's own
# message records the cost ("floored_margin 이하의 --margin은 더 이상 바를 움직이지 않는다").
# The flag stays on the command line anyway, because the recorded arms carried it.
#
# --keep-models all is the one flag that differs from the recorded arms (they used the default,
# best). It cannot move a score: prune_models runs in the report node, after holdout has already
# scored the winner (automl_agent/nodes/report.py). What it buys is that a later pre-registered
# experiment can re-score these runs' losing iterations without paying for the runs again — the
# same models docs/REPLAN-SEEDS.md needed and docs/RESULTS.md's arms cannot supply.
#
# The thread-id prefixes are suffixed with 2 (llm2-, nollm2-) because llm-adult-seed42 and the five
# beside it are recorded artifacts of RESULTS.md and nothing here may write into them.
# bench/prompt_redo.py reads the recorded directories as extra legs (code_shift, prompt_shift).
#
# The random arm is not re-drawn: bench/random_search.py never reads the goal, adjudication
# re-scores the stored model.joblib instead of refitting, and none of the thirty recorded seed-42
# draws carries a key 0ef38f8 changed the executor's handling of (bench/prompt_redo.py checks this
# and writes the answer into the verdict file).
#
# The six free arms run first, all of them. If anything about this pipeline is broken, it costs CPU
# to find out instead of Opus tokens — and no_llm2 - no_llm (code_shift) is computable from the free
# half alone, which is the pair that decides whether prompt_shift has one lever or two.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in the order bench/prompt_redo.py's DATASETS lists them. The metric is per dataset
# (house_sales is regression) and is fixed in bench/datasets.py, not chosen here.
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
