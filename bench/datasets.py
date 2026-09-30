"""The six public OpenML datasets the three arms are compared on.

Roles:

* Paths and seeds — where data and cards live, which seeds run.
* Dataset record — one benchmark row and its file paths.
* Dataset list — the six datasets and the judged four.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Role: paths and seeds ---------------------------------------------------------------

# Relative on purpose: cards are committed, run from repo root.
BENCH_DIR = Path("bench")
# CSVs are gitignored; only scripts, cards, results are committed.
DATA_DIR = BENCH_DIR / "data"
# Committed so readers see every arm got the same card.
CARD_DIR = BENCH_DIR / "cards"

# Judged seed runs all arms; noise seeds run cheap ones.
JUDGED_SEED = 42
NOISE_SEEDS: tuple[int, ...] = (43, 44)
CHEAP_SEEDS: tuple[int, ...] = (JUDGED_SEED, *NOISE_SEEDS)


# --- Role: dataset record ----------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """One benchmark row; ``name`` is the CSV stem and thread-id prefix."""

    name: str
    data_id: int
    target: str
    task: str
    metric: str
    note: str

    @property
    def csv_path(self) -> Path:
        return DATA_DIR / f"{self.name}.csv"

    @property
    def csv_arg(self) -> str:
        """The CSV path with forward slashes, safe to commit in a card."""
        return self.csv_path.as_posix()

    def card_path(self, seed: int) -> Path:
        """This dataset's card for one seed; cards are per seed."""
        return CARD_DIR / f"{self.name}-seed{seed}.json"


# --- Role: dataset list ------------------------------------------------------------------

# Fixed here so all three arms score the same.
BINARY: tuple[Dataset, ...] = (
    Dataset(
        name="adult",
        data_id=1590,
        target="class",
        task="binary",
        metric="balanced_accuracy",
        note="48,842행 · 수치 6 + 범주형 8 · 결측 0.95%(3열이 1% 초과) · 소수 클래스 23.9%",
    ),
    Dataset(
        name="bank-marketing",
        data_id=1461,
        target="Class",
        task="binary",
        metric="balanced_accuracy",
        note="45,211행 · 수치 7 + 범주형 9 · 결측 없음 · 소수 클래스 11.7%(MIMIC과 같은 불균형) "
        "· 열 이름이 V1..V16으로 익명화됨",
    ),
    Dataset(
        name="speeddating",
        data_id=40536,
        target="match",
        task="binary",
        metric="balanced_accuracy",
        note="8,378행 · 수치 61 + 범주형 59 · 결측 1.83%(35열이 1% 초과) · 소수 클래스 16.5% "
        "· field(259개)는 카디널리티 상한으로 버려짐",
    ),
    Dataset(
        name="spambase",
        data_id=44,
        target="class",
        task="binary",
        metric="balanced_accuracy",
        note="4,601행 · 수치 57 · 결측 없음 · 소수 클래스 39.4% · 가장 좁은 슬라이스",
    ),
)

MULTICLASS: tuple[Dataset, ...] = (
    Dataset(
        name="jungle-chess",
        data_id=41027,
        target="class",
        task="multiclass",
        metric="balanced_accuracy",
        note="44,819행 · 수치 6(범주형 0) · 결측 없음 · 3개 클래스 51.5/38.9/9.7% · "
        "집계에 섞지 않고 따로 관찰. roc_auc·pr_auc는 이 task에서 계산되지 않고"
        "(binary_only), f1·precision·recall은 macro 평균으로 나옵니다 — "
        "balanced_accuracy는 이진에서와 같은 정의(클래스별 recall의 평균)라서 "
        "이진 4개와 같은 열에 적을 수 있는 유일한 수입니다",
    ),
)

REGRESSION: tuple[Dataset, ...] = (
    Dataset(
        name="house_sales",
        data_id=42731,
        target="price",
        task="regression",
        metric="r2",
        note="21,613행 · 수치 21(범주형 0) · 결측 없음 · 회귀 · 집계에 섞지 않고 따로 관찰. "
        "id는 없고 날짜는 date_year/month/day로 이미 분해돼 있으며, zipcode(70개)는 CSV에서 "
        "정수로 읽혀 수치로 들어감 — 세 팔 모두에 똑같이 적용되는 조건",
    ),
)

ALL: tuple[Dataset, ...] = BINARY + MULTICLASS + REGRESSION

BY_NAME: dict[str, Dataset] = {item.name: item for item in ALL}

# Written out so new binary sets never shift it.
JUDGED: tuple[Dataset, ...] = BINARY
JUDGED_NAMES: frozenset[str] = frozenset(item.name for item in JUDGED)
