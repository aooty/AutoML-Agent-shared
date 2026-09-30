#!/usr/bin/env bash
# The 10 runs behind docs/ROWBUDGET.md, one fit each.
# Control arm runs in a worktree with empty row_budget.
set -u

export AUTOML_USE_BEDROCK=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")/../.."
REPO=$(pwd)
CONTROL=${CONTROL:-../test-rb-off}

ARMS=${ARMS:-"off on"}

if [ ! -d "$CONTROL" ]; then
  echo "대조군 worktree가 없습니다: $CONTROL"
  echo "만드는 방법 (프롬프트 절 하나만 지운 커밋을 얹습니다):"
  echo "  git worktree add $CONTROL -b rb-off-control HEAD"
  echo "  # $CONTROL/automl_agent/nodes/planning.py 의 \"row_budget\" 값을 빈 문자열로:"
  echo "  #   \"row_budget\": \"\",   (describe_row_budget(...) 호출을 대체)"
  echo "  cd $CONTROL && git commit -am '대조군: row budget 절을 비운다'"
  echo "  git diff HEAD~1 --stat   # 한 파일 한 줄이어야 합니다"
  exit 2
fi

# Link bench/data into the worktree; both read same bytes.
if [ ! -d "$CONTROL/bench/data" ]; then
  echo "대조군 worktree 에 데이터가 없습니다: $CONTROL/bench/data"
  echo "bench/data 는 gitignore 되어 있어 worktree 에 따라오지 않습니다. 같은 파일을 가리키게 하십시오:"
  echo "  cmd //c \"mklink /J ${CONTROL//\//\\\\}\\bench\\data bench\\data\"   # Windows"
  echo "  ln -s \"$REPO/bench/data\" \"$CONTROL/bench/data\"                    # POSIX"
  exit 2
fi

one() {
  local ds=$1 metric=$2 arm=$3 root=$4
  local tid="rb${arm}-${ds}-seed42"
  echo "=== ${tid} ($(date '+%H:%M:%S')) ==="
  ( cd "$root" && python -m automl_agent.main run \
      --dataset-card "bench/cards/${ds}-seed42.json" \
      --metric "$metric" \
      --goal-mode auto --margin 0.25 \
      --max-iterations 1 --time-budget-sec 3600 --seed 42 \
      --keep-models best \
      --artifacts-root "$REPO/bench/runs/artifacts" \
      --thread-id "$tid" )
  echo "--- ${tid} exit=$? ($(date '+%H:%M:%S')) ---"
}

# Both arms write to one artifacts root.
for pair in "spambase balanced_accuracy" "speeddating balanced_accuracy" \
            "house_sales r2" "bank-marketing balanced_accuracy" "adult balanced_accuracy"; do
  set -- $pair
  for arm in $ARMS; do
    case $arm in
      off) one "$1" "$2" off "$CONTROL" ;;
      on)  one "$1" "$2" on "$REPO" ;;
      *)   echo "알 수 없는 팔: $arm" >&2; exit 2 ;;
    esac
  done
done

echo "=== 열 번 다 끝났습니다 ($(date '+%H:%M:%S')) ==="
echo "계획 비교: 두 팔의 train/iter_01/train_config.json 의 hyperparams 를 대조하세요 (P1)"
