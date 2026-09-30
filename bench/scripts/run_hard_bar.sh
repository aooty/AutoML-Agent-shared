#!/usr/bin/env bash
# The twenty-four runs behind docs/HARD-BAR.md: the judged binary four x (bar65nollm + bar65llm) x
# seeds 42/43/44. Sequential on purpose: the arms share one checkpoints.sqlite, and two writers race
# for its lock.
#
# Why at all: docs/RESULTS.md and docs/PROMPT-REDO.md both measured llm - no_llm at fit counts of
# 1/1/2/1/1/1. Five of six runs reached goal_reached on their first attempt, so critic and the
# replanning path never ran — the numbers in those documents are about the first hyperparameter set,
# not about diagnosing and replanning. docs/REPLAN-SEEDS.md is the one condition where the loop
# actually iterated, and it is also the one place a loop-arm gain survived three seeds and a
# magnitude bar. It did that on one dataset. This widens it to four.
#
# One lever. Every flag below is bench/scripts/run_prompt_redo.sh's flag except --margin, which goes from
# 0.25 to 0.65. Prompt, code, cards, metrics, seeds, --max-iterations, --time-budget-sec and
# --keep-models are that script's. Changing anything else would make this two experiments instead of
# a re-measurement of one.
#
# Why 0.65 and not REPLAN-SEEDS' 0.5: 12fa74b floors the auto bar at the ranking ceiling, which
# makes --margin 0.5 inert on bank-marketing at all three seeds (floored_margin 0.526 / 0.531 /
# 0.523), and 0.6 clears the floor but lands at 0.8648 on that card at seed 42 — under the 0.8705 its
# first attempt already reached, so it would still stop at one fit. docs/HARD-BAR.md tabulates all
# twelve bars, computed by running goal.derive_threshold on the committed cards rather than
# remembered; the bar moves with the seed because the baseline is computed on the seed's own split.
# bench/hard_bar.py pins them (PREDICTED_BARS) and checks each run's actual goal.threshold against
# them (bar_check). Expect the reachability warning: goal.py discloses an out-of-reach bar and does
# not lower it, so the warning is the forced-iteration signal, not a mistake.
#
# --keep-models all is not a convenience: the observations re-score each run's iteration 1 on the
# test rows, and the default (best) deletes every non-winning iteration's model.joblib —
# bench/hard_bar.py refuses the cell rather than silently dropping the pair.
#
# The thread-id prefixes say bar65 rather than a numeric suffix because llm2-adult-seed42 and the
# five beside it are recorded artifacts of PROMPT-REDO.md and nothing here may write into them, and
# because an archive directory should say which bar its run faced. bench/hard_bar.py reads the
# seed-42 llm2/nollm2 directories as extra legs (bar_shift, rules_bar_shift) — one lever each: same
# prompt, same code, same card, same seed, only --margin differs.
#
# The random arm is not re-drawn: bench/random_search.py never reads the goal, so raising the bar
# cannot move a single draw, and the thirty recorded draws per (dataset, seed) are reused.
# bench/hard_bar.py reads the matched budget off each run's own fits rather than assuming five: an
# unreachable bar makes the loop run, not spend its whole budget. REPLAN-SEEDS.md's three runs are
# the only ones here that ever had five iterations against a bar they could not reach, and all three
# ended at stalled with critic_runs 3 (STALL_LIMIT is 2).
#
# The free arm runs first, all twelve of it. If anything about this pipeline is broken, it costs CPU
# to find out instead of Opus tokens — and rules_bar_shift (bar65nollm - nollm2) is computable from
# the free half alone. HARD-BAR.md's stop condition lives here: if any bar comes out away from the
# predicted table, or if bar65nollm records critic_runs 0 anywhere, stop before the paid half.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across thread
# counts, 0.99x the half-width of one paired verdict. Bedrock route because that is what every
# recorded llm arm used (route=bedrock, model=anthropic.claude-opus-5).
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in the order bench/hard_bar.py's DATASETS lists them (which is bench/datasets.py's
# JUDGED order). The metric is fixed per dataset in bench/datasets.py, not chosen here. house_sales
# and jungle-chess are absent on purpose: HARD-BAR.md does not widen RESULTS.md's binary-four
# denominator, and the diagnosis path this experiment is about is binary-only.
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
