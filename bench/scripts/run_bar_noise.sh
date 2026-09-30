#!/usr/bin/env bash
# The 14 runs behind docs/BAR-NOISE.md, serial (shared checkpoints.sqlite).
# Runs in a worktree at origin/lever-shape-guards, not main.
# Cost: about $66 paid; the two rules runs are free.
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

# Link bench/data into the worktree; both read same bytes.
if [ ! -d "$TREE/bench/data" ]; then
  echo "worktree 에 데이터가 없습니다: $TREE/bench/data"
  echo "bench/data 는 gitignore 되어 있어 worktree 에 따라오지 않습니다. 같은 파일을 가리키게 하십시오:"
  echo "  cmd //c \"mklink /J ${TREE//\//\\\\}\\bench\\data bench\\data\"   # Windows"
  echo "  ln -s \"$REPO/bench/data\" \"$TREE/bench/data\"                    # POSIX"
  exit 2
fi

# Cells: dataset and seed move together; see BAR-NOISE.md.
CELLS="adult:43:balanced_accuracy \
bank-marketing:44:balanced_accuracy \
spambase:44:balanced_accuracy \
speeddating:44:balanced_accuracy"

# N3 free cell; must match RULES_CELL in bar_noise.py.
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
    # Free half first: breakage costs CPU, not tokens.
    free)
      for r in 1 2; do
        one "bnoisenollm${r}" "$RULES_DATASET" "$RULES_SEED" "$RULES_METRIC" --no-llm
      done
      echo "=== 무료 두 번 끝났습니다 ($(date '+%H:%M:%S')) ==="
      echo "=== 유료 절반 전에 확인하십시오: python -m bench.bar_noise ${RULES_DATASET} ==="
      echo "===   N3 이 깨졌거나 바가 예측 표와 어긋나면 여기서 멈춥니다 ==="
      ;;
    # Finish one cell before the next, so partial runs count.
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
