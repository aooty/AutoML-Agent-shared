"""(데이터셋, 시드)마다 카드 하나를 만들고, 팔들이 필요한 내용이 들어 있는지 확인한다.

    python -m bench.cards                  # 데이터셋 전부 × 저렴한 시드 전부
    python -m bench.cards --seed 42        # 판정 대상 시드만
    python -m bench.cards adult spambase

**팔마다 만들지 않고 여기서 한 번만 만드는 이유.** ``run --data``는 파일을 직접 profile하므로 같은
데이터셋에 팔 셋이면 각자 자기 카드를 만든다. 그 카드들은 같을 것이다 — 시드가 고정이면 profiling은
결정적이다 — 그런데 일하는 단어가 "일 것이다"다. 한 번 만들어 세 팔 모두에 ``--dataset-card``로
넘기면 가정이 아니라 검사 가능한 사실이 되고, 모든 팔이 판정받는 목표 기준값도 서로 어긋날 때까지는
일치하는 유도 셋이 아니라 커밋된 파일 하나의 숫자 하나가 된다.

**exit status 말고 무엇을 확인하는가.** 만들어진 카드라고 벤치마크가 쓸 수 있는 카드는 아니다. 세
가지를 다시 읽고 거부한다:

* profiler가 감지한 ``task``가 ``datasets.py``의 선언과 맞는가 — 회귀 정답을 20개 클래스로 읽으면
  엉뚱한 실험에 대해 완벽히 유효한 카드가 만들어진다;
* 기준선이 있고 그 지표가 이 데이터셋을 판정하는 지표인가. ``--goal-mode auto``가 거기서 바를
  유도하고, 기준선 없는 카드는 이 데이터의 성질과 아무 상관 없는 지표별 기본값으로 조용히 내려앉는다;
* split 프로토콜이 이 시드를 기록했는가. ``automl_agent.nodes.profiling.assert_card_matches_protocol``
  이 어긋난 실행을 거부하는 필드다. 다섯 번 학습한 뒤보다 여기서 찾는 게 낫다.

기준선 점수를 출력한다. 결과가 아니라 바이고, 세 팔 모두가 넘어야 했던 것으로 ``RESULTS.md``에
들어간다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from bench.datasets import ALL, BY_NAME, CARD_DIR, CHEAP_SEEDS, Dataset

# ``command_profile``을 import하지 않고 subprocess로 부르는 이유는 저장소의 노드들과 같다:
# profiler는 pandas를 올리고 기준선을 학습시키는데 이 프로세스는 그 뒤에 JSON만 읽으면 된다.
# 덤으로 이 파일의 명령이 독자가 그대로 붙여 넣을 수 있는 명령이 된다.
PROFILE = [sys.executable, "-m", "automl_agent.main", "profile"]

# ``datasets.py``의 ``task``가 카드에서 무엇으로 돌아와야 하는가. 카드의 라벨은 registry의 둘보다
# 잘다(``classification``이 아니라 ``binary_classification``). registry가 뭉개는 이 구분
# (둘 다 ``TASK_CLASSIFICATION``으로 간다)이 ``roc_auc``/``pr_auc``를 아예 계산하는지, ``f1``이
# binary인지 macro인지를 정한다. 그래서 양쪽 다 선언한다 — 이항인데 multiclass로 돌아오면
# Critic의 이항 전용 진단이 그 데이터셋에서만 어긋나고, 다항인데 ``binary_classification``으로
# 돌아왔다면 앞단에서 클래스 둘이 합쳐진 것이다. 어느 방향이든 놀람이 아니라 거부가 된다.
CARD_TASK = {
    "binary": "binary_classification",
    "multiclass": "multiclass_classification",
    "regression": "regression",
}


def build(dataset: Dataset, seed: int, force: bool = False) -> int:
    card_path = dataset.card_path(seed)
    if card_path.exists() and not force:
        print(f"{dataset.name:16s} seed {seed}: 이미 있음 ({card_path.name}) — --force로 다시 만듭니다")
        return check(dataset, seed)

    CARD_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        *PROFILE,
        "--data",
        dataset.csv_arg,
        "--target",
        dataset.target,
        "--out",
        str(card_path),
        "--seed",
        str(seed),
        "--metric",
        dataset.metric,
        # 여기서 이름은 데이터셋의 공개 신분이다. 파일명이 prompt로 새지 않게 기본값을 두는
        # 임상 경로와 다르다. 이 이름들이 요점이다 — 독자가 다섯 중 어느 것에 대해 렌더된
        # prompt인지 알 수 있어야 한다.
        "--name",
        dataset.name,
    ]
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"{dataset.name:16s} seed {seed}: profile 실패 (exit {completed.returncode})")
        return 1
    normalise_path(dataset, seed)
    return check(dataset, seed)


def normalise_path(dataset: Dataset, seed: int) -> None:
    """카드의 ``data.path``를 슬래시로 고쳐 쓴다. 이 함수가 건드리는 유일한 필드다.

    ``profile``은 ``str(Path(args.data))``를 기록하고, Windows에서는 인자를 어떻게 적었든
    ``bench\\data\\adult.csv``가 된다. 이 카드들은 커밋되므로 Linux의 독자는 맞아 보이는 카드에서
    그 문자열을 ``read_csv``에 넘기고 파일을 놓친다. 슬래시는 어느 플랫폼에서든 ``pathlib``이 읽는다.

    profiler가 쓴 것을 고쳐 쓰는 일은 의심받을 만하니 범위를 밝힌다: 이건 비공개 ``data`` 블록이고,
    카드가 prompt에 닿기 전에 ``automl_agent.privacy.public_card``가 벗겨 내는 부분이며, 같은 파일을
    다르게 적은 것일 뿐이다. 집계도, 기준선도, 기준값도 건드리지 않는다.
    """
    card_path = dataset.card_path(seed)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    block = card.get("data")
    if isinstance(block, dict) and block.get("path"):
        block["path"] = str(block["path"]).replace("\\", "/")
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def check(dataset: Dataset, seed: int) -> int:
    """카드를 다시 읽어, 유효한 카드가 엉뚱한 카드일 수 있는 세 경로를 거부한다."""
    card: dict[str, Any] = json.loads(dataset.card_path(seed).read_text(encoding="utf-8"))
    problems: list[str] = []

    task = str(card.get("task") or "")
    if task != CARD_TASK[dataset.task]:
        problems.append(f"task가 {task!r}인데 {CARD_TASK[dataset.task]!r}를 기대했습니다")

    baseline = dict(card.get("baseline") or {})
    scores = dict(baseline.get("scores") or {})
    if not baseline:
        problems.append("baseline 블록이 없습니다 — --goal-mode auto가 이 데이터의 바를 못 세웁니다")
    elif dataset.metric not in scores:
        problems.append(f"baseline에 {dataset.metric}가 없습니다 (있는 것: {sorted(scores)})")

    protocol = dict(baseline.get("protocol") or {})
    if protocol.get("seed") != seed:
        problems.append(
            f"protocol.seed가 {protocol.get('seed')!r}입니다 — 이 카드로 seed {seed} 실행은 거부됩니다"
        )

    encoding = dict(card.get("encoding") or {})
    label = f"{dataset.name:16s} seed {seed}"
    if problems:
        for line in problems:
            print(f"{label}: {line}")
        return 1

    print(
        f"{label}: task {task} · 기준선 {dataset.metric} "
        f"{float(scores[dataset.metric]):.4f} · "
        f"수치 {encoding.get('numeric')} + one-hot {encoding.get('one_hot_columns')}열"
        f"→{encoding.get('one_hot_levels')} · 제외 "
        f"{encoding.get('dropped_high_cardinality') or []}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.cards", description=__doc__)
    parser.add_argument("names", nargs="*", help="데이터셋 이름 (생략하면 전부)")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help=f"카드를 만들 시드. 여러 번 쓸 수 있습니다 (생략하면 {list(CHEAP_SEEDS)})",
    )
    parser.add_argument("--force", action="store_true", help="이미 있는 카드도 다시 만듭니다")
    args = parser.parse_args(argv)

    names = args.names or [item.name for item in ALL]
    unknown = [name for name in names if name not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {unknown} — 아는 것은 {sorted(BY_NAME)}")
        return 2
    seeds = args.seeds or list(CHEAP_SEEDS)

    failed = 0
    for name in names:
        dataset = BY_NAME[name]
        if not dataset.csv_path.exists():
            print(f"{name}: {dataset.csv_path}가 없습니다 — 먼저 python -m bench.fetch {name}")
            failed += 1
            continue
        for seed in seeds:
            failed += build(dataset, seed, force=args.force)
    if failed:
        print(f"\n{failed}개 실패")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
