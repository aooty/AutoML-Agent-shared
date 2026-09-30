#!/usr/bin/env bash
# The twelve runs behind docs/WEIGHT-START.md: the judged binary four x seeds 42/43/44, rules
# fallback only. Sequential on purpose: the runs share one checkpoints.sqlite, and two writers race
# for its lock.
#
# Why at all: docs/WEIGHT-LEVER.md put a card ratio into the recorded rules winner's own config and
# recovered 67% / 84% / 114% of the delta docs/HARD-BAR.md had charged to the LLM arm. It said in
# its own words what it could not do — it never ran the loop, so it measured "is that gap the weight"
# rather than "what would the proposed code change produce". Commit d37217e made the change: the
# Critic's first class_weight rung comes from the card's imbalance ratio instead of from WEIGHT_STEP.
# These twelve runs are that code in the loop.
#
# One lever. Every flag below is bench/scripts/run_hard_bar.sh's flag for its free arm, unchanged: same
# cards, same metrics, same --margin 0.65, same --max-iterations, --time-budget-sec, --keep-models
# and --artifacts-root. The only difference between bar65nollmratio and the recorded bar65nollm is
# the commit the code is at. Changing anything else would make this two experiments instead of a
# measurement of one.
#
# --keep-models all is not a convenience: bench/weight_start.py re-scores this arm's iteration 1
# (ratio_replan_gain), and the default (best) deletes every non-winning iteration's model.joblib —
# the module refuses the cell rather than silently dropping the pair.
#
# No API key and no Bedrock env: --no-llm means the Critic runs heuristic_verdict, which is the
# branch d37217e changed. run_hard_bar.sh exported AUTOML_USE_BEDROCK=1 because half its runs were
# paid; none of these are, and this experiment costs zero tokens. The premise that lets the paid arm
# be reused instead of re-run is that all 37 of its recorded Critic verdicts have source == "llm", so
# the fallback this commit touched was never on its path — bench/weight_start.py's source_check
# re-checks that from the archive and WEIGHT-START.md makes a mismatch a stop condition.
#
# The paid arm bar65llm and the pre-change arm bar65nollm are NOT re-run and nothing here writes into
# their directories. bench/weight_start.py reads them as legs: prescription (bar65nollmratio -
# bar65nollm), headline (bar65llm - bar65nollmratio), and main (bar65llm - bar65nollm) reused as the
# denominator of condition 1's half bar and checked against HARD-BAR.md's published four places.
#
# Expect the reachability warning: goal.py discloses an out-of-reach bar and does not lower it, so
# the warning is the forced-iteration signal, not a mistake. The bars are HARD-BAR.md's twelve —
# same cards, same margin — and bench/weight_start.py checks each run's actual goal.threshold
# against hard_bar.PREDICTED_BARS rather than assuming.
#
# Stop conditions live here, all three from WEIGHT-START.md: if any cell's first class_weight
# prescription is not the pinned rung (adult it1 3.179, bank-marketing it1 7.547, speeddating it2
# 5.072, spambase none at all), or if any verdict's source is not "heuristic", or if any bar comes
# out away from the predicted table, the judgment is 판정 불가 and the criterion is not softened.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across thread
# counts, 0.99x the half-width of one paired verdict.
#
# Cost: CPU only. The recorded twelve of this arm spent 368 seconds of training in total, so expect
# around six minutes of fitting plus loop overhead.
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."

# name:metric, in the order bench/hard_bar.py's DATASETS lists them (which is bench/datasets.py's
# JUDGED order). The metric is fixed per dataset in bench/datasets.py, not chosen here. house_sales
# and jungle-chess are absent on purpose: the changed branch sits behind a SYMMETRIC_METRICS gate and
# is binary-only, and WEIGHT-START.md does not widen HARD-BAR.md's binary-four denominator.
SETS="adult:balanced_accuracy \
bank-marketing:balanced_accuracy \
speeddating:balanced_accuracy \
spambase:balanced_accuracy"

one() {
  local dataset=$1 metric=$2 seed=$3
  local tid="bar65nollmratio-${dataset}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  python -m automl_agent.main run \
    --dataset-card "bench/cards/${dataset}-seed${seed}.json" \
    --metric "$metric" \
    --goal-mode auto --margin 0.65 \
    --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
    --keep-models all \
    --artifacts-root bench/runs/artifacts \
    --thread-id "$tid" \
    --no-llm
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for seed in 42 43 44; do
  for pair in $SETS; do
    one "${pair%%:*}" "${pair##*:}" "$seed"
  done
done

echo "=== 열두 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
echo "=== 판정: python -m bench.weight_start ==="
