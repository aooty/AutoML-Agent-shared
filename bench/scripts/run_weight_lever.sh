#!/usr/bin/env bash
# The thirty-six fits behind docs/WEIGHT-LEVER.md: the judged binary four x three treatments x
# seeds 42/43/44. No API calls — every leg is one train.py invocation, so this costs CPU and
# nothing else.
#
# Why at all: docs/HARD-BAR.md forced the loop to iterate and still came out 구분되지 않음, one
# win in four. For that one win (speeddating) it wrote a post-hoc explanation — the rules fallback
# climbs class_weight from 1 by WEIGHT_STEP = 1.5 and only when the Critic picks that branch again,
# so it froze at 1.5 against a card asking for 5.07, while the LLM arm read the ratio off the card
# in one step. That document called the explanation post-hoc and left the check outside itself.
#
# One lever. Each leg is the recorded bar65nollm winner's own train_config.json with
# hyperparams.class_weight replaced and nothing else: same family, same remaining hyperparameters,
# same preprocessing, same rows, same seed, same metric. bench/weight_lever.py refuses a cell whose
# winner family does not publish class_weight (xgboost's lever is scale_pos_weight, and writing the
# other key would record a treatment that never happened) and refuses a cell whose recorded weight
# is not the rung RECORDED_WEIGHTS pinned before these fits existed.
#
# Three treatments, and the two controls are not decoration. `ratio` is the one the criterion is
# about — the ladder's starting point moved to the card's ratio. `balanced` is the spelling the LLM
# arm actually used, and it is *not* the same treatment at the same ratio: sklearn's balanced is
# n / (2 * bincount), so both weights move and logreg's fixed C then regularises relatively less.
# `refit` is the recorded value untouched, and without it the treatment deltas have no floor: A2
# measured 0.0026 of balanced_accuracy between two key-for-key identical xgboost fits, and whether
# hist_gbdt and logreg have such a floor is measured here for the first time.
#
# Nothing here writes into bench/runs/artifacts. bar65llm and bar65nollm are recorded artifacts of
# HARD-BAR.md; this reads their winners and compares against them, which also means the main delta
# is reused rather than re-measured and these are the same test rows a third comparison is being
# laid on. Both facts are written into the verdict file.
#
# Thread counts are pinned, and here it is load-bearing rather than hygienic: A2 measured a
# balanced_accuracy span of 0.0077 across thread counts, 0.99x the half-width of one paired
# verdict, and this experiment's refit control is a claim about what a refit does *not* move.
# bench/weight_lever.py pins them in the child environment too, so this export is the belt to that
# suspenders.
#
# Stop condition, from WEIGHT-LEVER.md: if the refit control moves further than 0.0026 on any of
# speeddating's three cells, no verdict is issued — that finding is larger than this experiment.
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$(dirname "$0")/../.."

echo "=== 서른여섯 번 학습 ($(date '+%H:%M:%S')) ==="
python -m bench.weight_lever --fit
fit_status=$?
echo "--- 학습 exit=${fit_status} ($(date '+%H:%M:%S')) ---"

if [ "$fit_status" -ne 0 ]; then
  echo "=== 학습에 실패한 칸이 있습니다 — 판정하지 않습니다 ==="
  echo "=== bench/runs/weightlever/fits.json을 보십시오 ==="
  exit "$fit_status"
fi

echo "=== 판정 ($(date '+%H:%M:%S')) ==="
python -m bench.weight_lever
echo "=== 끝났습니다 ($(date '+%H:%M:%S')) ==="
