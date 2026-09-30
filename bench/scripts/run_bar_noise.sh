#!/usr/bin/env bash
# The 14 runs behind docs/BAR-NOISE.md: four cells x three repeats of one configuration (paid),
# plus two free rules-arm repeats in one cell. Sequential on purpose: the runs share one
# checkpoints.sqlite, and two writers race for its lock.
#
# Why at all: docs/HARD-BAR.md judged twelve primary deltas against a magnitude bar of 0.0122 it
# did not measure. Its own text says so twice (:141, :218) — that number comes from
# docs/REPEATS.md, measured under the old prompt at --margin 0.25 on runs that fitted *once*.
# HARD-BAR forced the loop to iterate, so its runs fitted three to five times. The run-to-run term
# in that condition has never been measured, and HARD-BAR's own :312 reports adult's rules arm
# moving 0.0126 on the seed lever alone — larger than the bar it borrowed.
#
# Nothing moves but --thread-id, which is the checkpoint key and reaches no prompt. Every other
# flag below is bench/scripts/run_hard_bar.sh's flag, --margin 0.65 included. Changing anything else would
# make this a second experiment instead of a measurement of the first one's ruler.
#
# The baseline is not re-run. Every primary delta subtracts the recorded bar65nollm-<ds>-seed<seed>
# winner already in bench/runs/artifacts — same code, same card, same seed, same bar. A rules run
# with no LLM in it should be deterministic, and then the recorded run already *is* the repeat.
# "Should be" is an assumption nobody measured, so the free half below runs it twice more in one
# cell and bench/bar_noise.py's N3 checks the three points against SCORE_TOLERANCE.
#
# THE CODE IS NOT main. HARD-BAR's twenty-four runs came from a tree whose automl_agent/ ends at
# 0ef38f8 (HARD-BAR.md:173 pins that to the lever-shape-guards branch, which is why it was kept).
# Two commits have landed on automl_agent/ since: d37217e starts the class-weight ladder at the
# card's ratio — WEIGHT-START.md *measured* that changing a result (speeddating seed44 went -0.0520
# to +0.0326) — and 268e95d deletes unused paths, which is behaviour-preserving but is also a
# claim, and the term this experiment measures is around 0.005. So the runs happen in a worktree at
# origin/lever-shape-guards. bench/cards/ is byte-identical between the two trees, so the bar is
# the same bar; bench/bar_noise.py checks each run's goal.threshold against hard_bar.PREDICTED_BARS
# rather than trusting that.
#
# A worktree and not a CLI flag, for the reason bench/scripts/run_rowbudget.sh gives: an experimental flag
# left in the product is a permanent door, and a worktree leaves nothing behind but this script and
# the verdict. Nothing here can verify the tree — run_config.json records no commit — so it is a
# claim this script and the document make, and the branch is what keeps it true.
#
# Thread counts are pinned because A2 measured a balanced_accuracy span of 0.0077 across thread
# counts, 1.5x the number N1's threshold sits at. Bedrock route because that is what every recorded
# llm arm used (route=bedrock, model=anthropic.claude-opus-5).
#
# Cost: ~$66 for the paid twelve, measured off HARD-BAR's own archives rather than guessed
# ($3.85 / $5.35 / $5.87 / $6.98 per run for the four cells at $15/M input, $75/M output).
# BAR-NOISE.md pre-registered a $85 ceiling. The free two cost nothing but CPU.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."
REPO=$(pwd)
TREE=${TREE:-../test-bar65}
BRANCH=${BRANCH:-origin/lever-shape-guards}

HALVES=${HALVES:-"free paid"}

if [ ! -d "$TREE" ]; then
  echo "HARD-BAR가 돈 코드의 worktree가 없습니다: $TREE"
  echo "만드는 방법 (--detach: 이 커밋에 브랜치를 얹지 않습니다):"
  echo "  git worktree add --detach $TREE $BRANCH"
  exit 2
fi

# 카드의 data.path 는 cwd 기준 상대경로이고, 실행은 worktree 에서 돕니다. bench/data 는
# gitignore 되어 있으므로 worktree 에는 없습니다 — 없으면 열네 실행이 전부 data_issue 로 죽고,
# ROWBUDGET 때 실제로 한 번 그렇게 태웠습니다. 복사가 아니라 junction 인 이유: 두 조건이 같은
# 바이트를 읽는 것이 이 실험의 공정성 주장입니다.
if [ ! -d "$TREE/bench/data" ]; then
  echo "worktree 에 데이터가 없습니다: $TREE/bench/data"
  echo "bench/data 는 gitignore 되어 있어 worktree 에 따라오지 않습니다. 같은 파일을 가리키게 하십시오:"
  echo "  cmd //c \"mklink /J ${TREE//\//\\\\}\\bench\\data bench\\data\"   # Windows"
  echo "  ln -s \"$REPO/bench/data\" \"$TREE/bench/data\"                    # POSIX"
  exit 2
fi

# 셀: 데이터셋과 시드가 함께 움직입니다. HARD-BAR 의 데이터셋 판정이 걸린 시드를 카드마다 하나씩
# 골랐고, 고른 이유는 BAR-NOISE.md 의 표에 있습니다. 지표는 bench/datasets.py 가 카드마다
# 고정하는 것이고 여기서 고르는 것이 아닙니다.
CELLS="adult:43:balanced_accuracy \
bank-marketing:44:balanced_accuracy \
spambase:44:balanced_accuracy \
speeddating:44:balanced_accuracy"

# N3 의 무료 대조가 도는 셀. bar_noise.py 의 RULES_CELL 과 같아야 합니다.
RULES_DATASET=adult
RULES_SEED=43
RULES_METRIC=balanced_accuracy

one() {
  local arm=$1 ds=$2 seed=$3 metric=$4 extra=${5:-}
  local tid="${arm}-${ds}-seed${seed}"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  ( cd "$TREE" && python -m automl_agent.main run \
      --dataset-card "bench/cards/${ds}-seed${seed}.json" \
      --metric "$metric" \
      --goal-mode auto --margin 0.65 \
      --max-iterations 5 --time-budget-sec 3600 --seed "$seed" \
      --keep-models all \
      --artifacts-root "$REPO/bench/runs/artifacts" \
      --thread-id "$tid" \
      $extra )
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

for half in $HALVES; do
  case $half in
    # 무료가 먼저입니다. 파이프라인이 깨져 있으면 Opus 토큰이 아니라 CPU 로 알게 되고,
    # N3 은 이 절반만으로 판정됩니다 — 그리고 N3 이 깨지면 유료 열두 번을 돌리기 전에
    # 설계를 다시 해야 합니다 (BAR-NOISE.md 의 멈추는 조건).
    free)
      for r in 1 2; do
        one "bnoisenollm${r}" "$RULES_DATASET" "$RULES_SEED" "$RULES_METRIC" --no-llm
      done
      echo "=== 무료 두 번 끝났습니다 ($(date '+%H:%M:%S')) ==="
      echo "=== 유료 절반 전에 확인하십시오: python -m bench.bar_noise ${RULES_DATASET} ==="
      echo "===   N3 이 깨졌거나 바가 예측 표와 어긋나면 여기서 멈춥니다 ==="
      ;;
    # 셀 하나를 끝내고 다음 셀로 갑니다. 반복을 바깥 루프로 두면 중간에 멈췄을 때 어느 셀도
    # 3회를 못 채우고, N1 의 span 은 3회가 다 있어야 그 span 입니다 — 이렇게 두면 중단된
    # 실행도 끝난 셀만큼은 판정됩니다.
    paid)
      for cell in $CELLS; do
        IFS=: read -r ds seed metric <<<"$cell"
        for r in 1 2 3; do
          one "bnoise${r}" "$ds" "$seed" "$metric"
        done
        echo "=== ${ds} 시드 ${seed} 세 반복 끝났습니다 ($(date '+%H:%M:%S')) ==="
      done
      ;;
    *)
      echo "알 수 없는 절반: $half" >&2
      exit 2
      ;;
  esac
done

echo "=== 끝났습니다 ($(date '+%H:%M:%S')) ==="
echo "판정: python -m bench.bar_noise"
