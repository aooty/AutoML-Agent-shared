#!/usr/bin/env bash
# The six runs behind docs/REPLAN-SEEDS.md: speeddating x (hard_nollm2 + hard_llm2) x seeds
# 42/43/44. Sequential on purpose: the arms share one checkpoints.sqlite, and two writers race for
# its lock.
#
# Why speeddating alone, and why these three seeds: REPLAN.md measured a replanning gain of +0.0935
# there and nowhere else, on one seed, under the prompt that predated docs/REGISTRY-GAP.md. This
# asks whether that one number survives a seed change and the new prompt. Seed 42 is re-run because
# separating "the prompt changed" from "the seed changed" needs a new-prompt run at the recorded
# seed — otherwise every delta carries both levers.
#
# The thread-id prefixes are suffixed with 2 (hardllm2-, hardnollm2-) because hardllm-speeddating-
# seed42 is a recorded artifact of REPLAN.md and nothing here may write into it. bench/replan_seeds.py
# reads the recorded directories as extra legs (code_shift, prompt_shift).
#
# --keep-models all is not a convenience: the pre-registered criterion re-scores each run's
# iteration 1 on the test rows, and the default (best) deletes every non-winning iteration's
# model.joblib — bench/replan_seeds.py refuses the seed rather than silently dropping the pair.
#
# --margin 0.5 is REPLAN.md's condition, kept identical. The bars were pre-computed on the committed
# cards and written into the pre-registration: 0.8342 / 0.8399 / 0.8403 for seeds 42 / 43 / 44,
# against the recorded hard_nollm arm's best val of 0.6998 / 0.7327 / 0.7080. All three are out of
# reach, which is what makes the loop run.
#
# The free arm runs first, all three of it. If anything about this pipeline is broken, it costs CPU
# to find out instead of Opus tokens.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across thread
# counts, 0.99x the half-width of one paired verdict. Bedrock route because that is what every
# recorded llm arm used (route=bedrock, model=anthropic.claude-opus-5).
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
