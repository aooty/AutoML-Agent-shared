"""Build one card per dataset and seed, then check its contents.

Roles:

* Constants — the profile command and expected card task names.
* Build — run the profiler and fix the data path.
* Check — reject cards with wrong task, baseline, or seed.
* CLI — pick datasets and seeds, build each card.

Usage:
    python -m bench.cards
    python -m bench.cards --seed 42
    python -m bench.cards adult spambase
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from bench.datasets import ALL, BY_NAME, CARD_DIR, CHEAP_SEEDS, Dataset

# --- Role: constants ---------------------------------------------------------------------

# Subprocess keeps pandas out of this process.
PROFILE = [sys.executable, "-m", "automl_agent.main", "profile"]

# A task mismatch here is a refusal, not a surprise.
CARD_TASK = {
    "binary": "binary_classification",
    "multiclass": "multiclass_classification",
    "regression": "regression",
}


# --- Role: build -------------------------------------------------------------------------


def build(dataset: Dataset, seed: int, force: bool = False) -> int:
    """Profile one dataset at one seed into its card, then check it."""
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
        # Public name on purpose, unlike the clinical path.
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
    """Rewrite only the private ``data.path`` with forward slashes."""
    card_path = dataset.card_path(seed)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    block = card.get("data")
    if isinstance(block, dict) and block.get("path"):
        block["path"] = str(block["path"]).replace("\\", "/")
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


# --- Role: check -------------------------------------------------------------------------


def check(dataset: Dataset, seed: int) -> int:
    """Reread the card; reject wrong task, missing baseline, or wrong seed."""
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


# --- Role: CLI ---------------------------------------------------------------------------


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
