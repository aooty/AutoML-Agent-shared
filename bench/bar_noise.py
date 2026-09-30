"""``docs/BAR-NOISE.md`` 판정: ``--margin 0.65``에서 같은 설정 하나가 얼마나 움직이는가.

``docs/HARD-BAR.md``는 주 Δ 열두 개를 내놓고, 자기가 재지 않은 크기 바 0.0122에 대고 판정했다.
그 문서 자신이 두 번 그렇게 적는다(:141, :218): 그 수는 ``docs/REPEATS.md``에서 왔고, 그 문서는 낡은
프롬프트로 ``--margin 0.25``에서 학습이 **한 번** 도는 실행에서 쟀다. HARD-BAR는 루프를 강제로 돌렸으니
그쪽 실행은 세 번에서 다섯 번 학습했다. *그* 조건의 실행 간 항은 아무도 재지 않았다. 이 모듈이 잰다.

사전 등록은 ``docs/BAR-NOISE.md``이고, **이 파일은 그 실행이 하나도 생기기 전에 커밋됐다** — 다른
판정기 넷이 따르는 것과 같은 규칙이고 이유도 같다: 숫자를 보고 쓴 스크립트는 그 숫자에 맞출 수 있고,
쓰면서 얼마나 조심했든 그게 아니라는 증명은 되지 않는다. 증명은 git 순서다.

하지 않는 것. ``HARD-BAR.md``의 판정을 다시 계산하지 않고 자기 파일을 쓴다
(``bench/runs/paired/bar-noise.json``). 관계는 ``REPEATS.md``가 ``RESULTS.md``에 대해 갖는 관계와 같다.
기록된 ``bar65llm`` 실행에 반복을 짝지우지도 않는다 — 그 실행이 코드·카드·시드를 공유하긴 하지만, 그
네 실행은 똑같이 정당한 점 네 개이고, 답을 가장 넓게 혹은 가장 좁게 만드는 쪽을 고르는 것은 숫자를 보고
고르는 것이다.

baseline은 다시 돌리지 *않는다*. 여기 모든 주 Δ는 기록된 ``bar65nollm-<dataset>-seed<seed>`` 우승자를
뺀다 — 같은 코드, 같은 카드, 같은 시드, 같은 바. LLM이 없는 규칙 실행은 결정적이어야 하고 그렇다면
기록된 실행이 이미 반복 *그 자체*이기 때문이다. "여야 한다"는 측정이 뒤에 없는 가정이라,
:data:`RULES_CELL`이 두 번 더 돌리고 N3가 확인한다. 그 절반은 API 토큰이 0이고, 그래서 먼저 돈다.

코드는 ``main``이 아니다. 이 실행들은 ``origin/lever-shape-guards``의 worktree에서 돌고, 그
``automl_agent/``가 HARD-BAR의 24실행이 쓴 나무다(``HARD-BAR.md:173``). 그 뒤 동작에 관계된 커밋 둘이
들어왔고 그중 하나는 결과를 바꾸는 것으로 측정됐다(``WEIGHT-START.md``).
``bench/scripts/run_bar_noise.sh``가 그 설정을 한다. 여기서는 확인할 수 없다 — ``run_config.json``에
커밋이 기록되지 않는다 — 그래서 runner와 문서가 하는 주장이고, 브랜치가 그걸 참으로 유지한다.

팔들이 돈 것처럼 ``OMP_NUM_THREADS=1``로 돌린다. 값은 출력에 기록된다.

사용법:
    python -m bench.bar_noise                   # 셀 네 개
    python -m bench.bar_noise adult             # 데이터셋 하나의 셀
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME
from bench.hard_bar import BAR_TOLERANCE, PREDICTED_BARS
from bench.hard_bar import MARGIN as BAR_MARGIN
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
    Delta,
    Refusal,
    Scored,
    Winner,
    loop_winner,
    paired_delta,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.repeats import check_one_environment, pinned, recorded_threads
from bench.replan import run_shape

# 셀 네 개, 판정된 카드마다 하나: 그 데이터셋에 대한 ``HARD-BAR.md``의 판정이 정해진 시드.
# 여기 리터럴로 못박는 이유는 사전 등록된 범위가 그런 것이기 때문이다 — 판정 파일에서 유도하면 그
# 파일이 움직일 때마다 함께 움직이고, 요점은 이 분모가 어떤 반복보다도 먼저 정해졌다는 것이다.
# 각각의 이유는 ``BAR-NOISE.md``가 표로 적는다: adult-43과 spambase-44는 구간이 0 아래로 떨어진 두 셀,
# bank-marketing-44는 0을 벗어난 유일한 bank 셀, speeddating-44는 이김 셋 중 *가장 작은* 것.
CELLS: tuple[tuple[str, int], ...] = (
    ("adult", 43),
    ("bank-marketing", 44),
    ("spambase", 44),
    ("speeddating", 44),
)

# 명령줄이 ``--thread-id`` 하나만 다른 실행 셋. 그건 checkpoint 키이고 어떤 프롬프트에도 닿지
# 않는다. 이름이 아니라 번호인 이유는 여기 어느 팔도 대조군이 아니기 때문이다 — 대조군은 기록된
# 규칙 실행이다.
REPEATS: tuple[int, ...] = (1, 2, 3)

# 모든 주 Δ가 빼는 기록된 규칙 팔. 다시 돌리지 않고 HARD-BAR 자신의 아카이브 디렉터리에서
# 읽는다. 모듈 docstring 참고.
BASELINE = "bar65nollm"

# 결정성 검사. 셀 하나인 이유는 API 비용은 0이지만 CPU 비용은 0이 아니고, 한 셀에서 비결정적인
# 규칙 팔은 네 셀 전부를 읽는 방식을 바꾸기 때문이다.
RULES_CELL: tuple[str, int] = ("adult", 43)
RULES_REPEATS: tuple[int, ...] = (1, 2)

# ``HARD-BAR.md``가 짐으로 판정한 셀들 — N2가 묻는 대상. ``CELLS``와 같은 이유로 리터럴이다:
# 다시 검토받는 주장이고, 검토보다 먼저 못박혀 있어야 한다.
LOST_CELLS: tuple[tuple[str, int], ...] = (("adult", 43), ("spambase", 44))

# N1의 배수. ``REPEATS.md``의 R1이 쓴 것과 같아서 두 답이 한 자 위에 놓인다.
N1_THRESHOLD = 0.5

# N4가 span 중앙값 옆에 적는 것: ``REPEATS.md``의 값이고 HARD-BAR가 수입한 수. 다시 계산하지 않고
# 상수로 두는 이유는 그게 *공표된* 수이고, 그 줄의 요점이 독자가 그 문서에 대고 확인할 수 있다는
# 것이기 때문이다. 관찰뿐이다 — 두 측정은 카드 집합·시드·바·코드가 다르고, N4는 둘을 나누지 않고
# 그렇게 적는다.
REPEATS_NOISE_FLOOR = 0.0122
REPEATS_NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트 · --margin 0.25 · 학습 1회)"

# N1이 비교 대상으로 삼는 반폭을 읽어 오는 곳. HARD-BAR의 파일인 이유는 N1이 이름을 대는 양이
# 그 문서가 실제로 공표한 구간의 폭이기 때문이다.
HARD_BAR_VERDICT = OUT_DIR / "hard-bar.json"

# ``bench/hard_bar.py`` 참고: 재추출 시드는 실행 시드가 아니다. 파일의 모든 짝이 한 수열에서
# 나오도록, 그리고 판정 파일마다 ``seed`` 하나를 읽는 ``bench/recheck.py``가 전부 다시 계산할 수
# 있도록 못박는다.
RESAMPLE_SEED = 42


def cell_key(dataset: str, seed: int) -> str:
    """판정 항목의 이름. 양쪽 다 선택의 일부이므로 둘 다 들어간다."""
    return f"{dataset}-seed{seed}"


def repeat_arm(repeat: int) -> str:
    return f"bnoise{repeat}"


def rules_arm(repeat: int) -> str:
    return f"bnoisenollm{repeat}"


def run_dir(prefix: str, dataset: str, seed: int) -> Path:
    """``bench/scripts/run_bar_noise.sh``가 실행 하나를 둔 곳. thread-id가 디렉터리 이름이다."""
    return ARTIFACTS_DIR / f"{prefix}-{dataset}-seed{seed}"


# --------------------------------------------------------------------------- #
# N1이 비교 대상으로 삼는 반폭
# --------------------------------------------------------------------------- #


def recorded_half_widths(path: Path | None = None) -> dict[str, float]:
    """셀마다, ``HARD-BAR.md``의 주 Δ 반폭.

    이 모듈에 옮겨 적지 않고 판정 파일에서 읽는다. 네 수는 ``BAR-NOISE.md``의 산문에 있어서 독자가
    문턱이 정해진 시점의 값을 볼 수 있지만, 계산은 파일을 읽는다 — 손으로 복사한 상수와 공표된 판정은
    한 수를 두 곳에 두는 것이고, 그건 어긋난다.

    주 짝만, 이름으로(``bar65llm`` − ``bar65nollm``), 그리고 중앙값이 아니라 셀마다 하나:
    데이터셋마다 주 짝이 여러 개였던 ``REPEATS.md``의 R1과 달리, 여기 각 셀에는 폭이 문제되는 구간이
    정확히 하나다.
    """
    verdict = read_json_object(path or HARD_BAR_VERDICT)
    if verdict is None:
        return {}
    widths: dict[str, float] = {}
    for entry in verdict.get("datasets") or []:
        for pair in entry.get("pairs") or []:
            if (
                pair.get("kind") == "primary"
                and pair.get("a") == "bar65llm"
                and pair.get("b") == BASELINE
                and isinstance(pair.get("ci_low"), (int, float))
                and isinstance(pair.get("ci_high"), (int, float))
            ):
                widths[str(entry.get("dataset"))] = (
                    float(pair["ci_high"]) - float(pair["ci_low"])
                ) / 2
    return widths


# --------------------------------------------------------------------------- #
# 셀 하나
# --------------------------------------------------------------------------- #


def _collect(dataset: str, seed: int, metric: str, out: DatasetVerdict) -> dict[str, Winner]:
    """반복들, 기록된 baseline, 그리고 한 셀에서는 무료 결정성 다리까지."""
    winners: dict[str, Winner] = {}
    missing: list[str] = []
    for repeat in REPEATS:
        arm = repeat_arm(repeat)
        winner = loop_winner(arm, run_dir(arm, dataset, seed), metric)
        if winner is None:
            missing.append(arm)
        else:
            winners[arm] = winner
    if missing:
        # 거부가 아니라 기록. "셋 다 아니면 없음"은 양을 계산하는 곳에서 강제된다: 점이 셋보다
        # 적으면 ``span``이 ``None``을 돌려주고 ``primary_deltas``가 구간을 셋보다 적게 내므로,
        # 여기 도움 없이도 N1은 부분 판정, N2는 판정 안 함으로 보고한다. 여기서 셀을 거부하면 아래서
        # 모으는 규칙 다리까지 버리게 되고, 그건 N3가 답하는 재료다 — 유료 12실행 *전에* 돌리라고
        # BAR-NOISE.md가 사전 등록한 검사다. 여기서 거부하면 그 검사에 닿을 수 없었다.
        out.missing_arms.extend(missing)

    baseline = loop_winner(BASELINE, run_dir(BASELINE, dataset, seed), metric)
    if baseline is None:
        out.missing_arms.append(BASELINE)
        raise Refusal(
            f"{cell_key(dataset, seed)}: 기록된 {BASELINE} 실행이 없습니다 — "
            "주 Δ의 baseline은 HARD-BAR의 규칙 팔이고 여기서 다시 돌리지 않습니다"
        )
    winners[BASELINE] = baseline

    if (dataset, seed) == RULES_CELL:
        for repeat in RULES_REPEATS:
            arm = rules_arm(repeat)
            winner = loop_winner(arm, run_dir(arm, dataset, seed), metric)
            # 없는 것은 거부가 아니다: N3가 "판정 안 함"으로 보고하고 유료 절반의 숫자는 그대로
            # 선다. 이 실행들은 무료라서 일어나지 않아야 하지만 — 무료 다리 하나가 없다고 유료 측정
            # 네 셀을 조용히 지우는 쪽이 더 나쁜 규칙이다.
            if winner is None:
                out.missing_arms.append(arm)
            else:
                winners[arm] = winner
    return winners


def _pairs_to_take(scored: dict[str, Scored]) -> list[tuple[str, str, str]]:
    """이 셀이 낼 수 있는 비교. ``BAR-NOISE.md``가 나열한 순서로."""
    pairs: list[tuple[str, str, str]] = []
    repeats = [repeat_arm(r) for r in REPEATS if repeat_arm(r) in scored]
    # N2의 Δ: 반복마다, HARD-BAR가 쓴 것과 같은 기록된 baseline에 대고.
    for arm in repeats:
        pairs.append((arm, BASELINE, "primary"))
    # N1의 재료. span만이 아니라 구간으로도 낸다: 반복 대 반복은 공유된 행에서의 짝지은 차이라
    # 구간을 갖고, ``REPEATS.md``의 R2가 그런 짝 12개 중 4개가 0을 벗어난 것을 찾았다.
    for i, a in enumerate(repeats):
        for b in repeats[i + 1 :]:
            pairs.append((a, b, "repeat"))
    # N3: 규칙 팔을 자기 자신에 대고. 모든 행에서 부호가 같게 읽히도록 이름이 앞인 쪽을 먼저.
    rules = [BASELINE, *(rules_arm(r) for r in RULES_REPEATS if rules_arm(r) in scored)]
    if len(rules) > 1:
        for i, a in enumerate(rules):
            for b in rules[i + 1 :]:
                pairs.append((a, b, "rules_repeat"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    metric = BY_NAME[dataset].metric
    winners = _collect(dataset, seed, metric, out)

    scored: dict[str, Scored] = {}
    threads: dict[str, dict[str, Any] | None] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        gap = abs(result.test_score - winner.recorded_test)
        if gap > SCORE_TOLERANCE:
            raise Refusal(
                f"{name}: 다시 유도한 test 점수 {result.test_score:.6f}가 기록된 "
                f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
            )
        scored[name] = result
        threads[name] = recorded_threads(winner)

    fingerprints = {name: s.fingerprint for name, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{name}={fp[:12]}" for name, fp in sorted(fingerprints.items()))
        raise Refusal(
            f"{cell_key(dataset, seed)}: 팔마다 test 지문이 다릅니다 ({listing}) — "
            "짝지을 행이 아닙니다"
        )
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{cell_key(dataset, seed)}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )
    # 두 실행이 서로 다른 스레드 상태에서 학습했음이 증명되면 올린다. A2가 스레드 수에 걸친 span을
    # 0.0077로 쟀고 그건 N1의 문턱이 앉은 수의 1.5배다 — 스레드 수가 움직인 반복은 그걸 재게 된다.
    check_one_environment(threads, cell_key(dataset, seed))

    shapes = {name: run_shape(w.directory.parent.parent) for name, w in winners.items()}
    any_scored = next(iter(scored.values()))
    out.test_rows = any_scored.n_rows
    out.test_fingerprint = any_scored.fingerprint
    for name, s in scored.items():
        out.arms[name] = {
            "label": s.winner.label,
            "fits": s.winner.fits,
            "val_score": s.winner.val_score,
            "test_recorded": s.winner.recorded_test,
            "test_reproduced": s.test_score,
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            # 셀마다가 아니라 실행마다: 이 필드의 요점이 두 실행이 여기서 다를 수 있다는 것이고,
            # 셀 수준 사본은 그중 하나만 기록할 수 있다.
            "threads": threads[name],
            "threads_pinned": pinned(threads[name]),
            **shapes[name],
        }

    for a_name, b_name, kind in _pairs_to_take(scored):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # 중간에 거부가 나면 다리를 남기지 않도록 맨 마지막에 — ``bench/paired.py``와 같은 규칙.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """셀 하나. ``paired.py``처럼 거부는 짝을 비운다."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=cell_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --------------------------------------------------------------------------- #
# 셀 하나 되읽기
# --------------------------------------------------------------------------- #


def repeat_scores(result: DatasetVerdict) -> list[float]:
    """반복들의 다시 유도한 test 점수. 반복 순서대로. baseline과 규칙 다리는 제외."""
    scores: list[float] = []
    for repeat in REPEATS:
        arm = result.arms.get(repeat_arm(repeat)) or {}
        value = arm.get("test_reproduced")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append(float(value))
    return scores


def span(result: DatasetVerdict) -> float | None:
    """반복들 test 점수의 최대−최소. 사전 등록된 span이 아니면 ``None``."""
    if result.verdict != "ok":
        return None
    scores = repeat_scores(result)
    if len(scores) < len(REPEATS):
        return None
    return max(scores) - min(scores)


def primary_deltas(result: DatasetVerdict) -> list[Delta]:
    """반복마다 기록된 baseline에 대고. 반복 순서대로."""
    order = {repeat_arm(r): i for i, r in enumerate(REPEATS)}
    found = [p for p in result.pairs if p.kind == "primary" and p.b == BASELINE and p.a in order]
    return sorted(found, key=lambda p: order[p.a])


def shape_agreement(result: DatasetVerdict) -> dict[str, Any]:
    """반복 셋이 루프를 같은 방식으로 썼는가.

    기준이 아니다 — ``BAR-NOISE.md``는 진술 넷을 사전 등록했고 이건 그중 하나가 아니다. 출력에 있는
    이유는, 반복들이 서로 다른 학습 횟수에서 멈춘 셀은 실행 간 항에 "루프가 다른 횟수로 돌았다"가
    섞인 셀이고, 그건 같은 span을 실질적으로 다르게 읽는 것이기 때문이다. 판정하지 않고 기록한다.
    """
    fields = ("fits", "stop_reason", "critic_runs")
    per_arm = {
        repeat_arm(r): {f: (result.arms.get(repeat_arm(r)) or {}).get(f) for f in fields}
        for r in REPEATS
        if repeat_arm(r) in result.arms
    }
    distinct = {json.dumps(row, sort_keys=True) for row in per_arm.values()}
    # 반복이 다 있지 않으면 ``True``가 아니라 ``None``: 비교한 것이 없는데 판정 파일에 공허한
    # "같음"이 적히면 실제로 한 관찰처럼 읽힌다.
    agree = len(distinct) <= 1 if len(per_arm) == len(REPEATS) else None
    return {"per_arm": per_arm, "agree": agree}


# --------------------------------------------------------------------------- #
# 각 실행이 마주한 바
# --------------------------------------------------------------------------- #


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """실행마다의 ``goal.threshold``를 ``hard_bar.PREDICTED_BARS``에 대고 비교한다.

    표를 다시 적지 않고 그 모듈에서 import한다: "같은 바가 아니면 같은 조건이 아니다"가
    ``BAR-NOISE.md``의 정지 조건이고, 그건 HARD-BAR가 예측하고 받은 바를 뜻한다. 그 열두 수를 여기
    두 번째로 복사하면 첫 번째와 달라질 수 있고, 그러면 검사가 마지막으로 고친 사본에 대고 이뤄진다.
    """
    rows: list[dict[str, Any]] = []
    for dataset, seed in CELLS:
        result = results.get((dataset, seed))
        if result is None:
            continue
        predicted = PREDICTED_BARS[(dataset, seed)]
        for arm, values in result.arms.items():
            actual = values.get("goal_threshold")
            # 실행이 문턱을 기록하지 않았으면 ``None``. 일치로 읽혀서는 안 된다.
            known: float | None = (
                float(actual)
                if isinstance(actual, (int, float)) and not isinstance(actual, bool)
                else None
            )
            off = abs(known - predicted) if known is not None else None
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "arm": arm,
                    "predicted_bar": predicted,
                    "actual_bar": actual,
                    "off_by": off,
                    "matches": off is not None and off <= BAR_TOLERANCE,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# 사전 등록된 진술
# --------------------------------------------------------------------------- #


def n1(
    results: dict[tuple[str, int], DatasetVerdict],
    half_widths: dict[str, float],
    threshold: float = N1_THRESHOLD,
) -> dict[str, Any]:
    """N1 판정. ``docs/BAR-NOISE.md``에서 인용:

    > 네 셀에서 반복 3회의 test 점수 span(최대−최소)의 **중앙값**을, **같은 네 셀**의
    > ``HARD-BAR.md`` 주 Δ 반폭 중앙값과 비교한다. span 중앙값이 반폭 중앙값의 **0.5배를
    > 넘으면**, ``HARD-BAR.md``의 모든 주 Δ는 "한 실행의 값"이 아니라 "이 크기의 항이 섞인 값"
    > 으로 다시 읽어야 한다고 적는다.

    두 중앙값은 *같은* 셀들에서 취한다 — span과 공표된 반폭이 둘 다 있는 셀들. 서로 다른 집합에서
    취하면 좁은 셀들의 중앙값을 넓은 셀들의 중앙값에 대고 비교하면서 그 비를 측정이라 부르게 된다.

    못박은 넷에 못 미치는 범위는 답이 아니라 부분이라고 이름을 붙인다. N1에는 선호하는 방향이 없다:
    작은 span은 작은 span으로 보고되고, 그건 HARD-BAR의 짐 둘을 단단하게 만드는 결과 — 이 저장소에
    가장 불리한 읽기 — 이며 같은 문장이 그걸 낸다.
    """
    per_cell: dict[str, dict[str, Any]] = {}
    for dataset, seed in CELLS:
        key = cell_key(dataset, seed)
        result = results.get((dataset, seed))
        value = None if result is None else span(result)
        width = half_widths.get(key)
        per_cell[key] = {
            "span": value,
            "half_width": width,
            "ratio": (value / width) if (value is not None and width) else None,
        }
    usable = [
        key
        for key in (cell_key(d, s) for d, s in CELLS)
        if per_cell[key]["span"] is not None and per_cell[key]["half_width"] is not None
    ]
    missing = [key for key in (cell_key(d, s) for d, s in CELLS) if key not in usable]
    out: dict[str, Any] = {
        "threshold": threshold,
        "expected_cells": [cell_key(d, s) for d, s in CELLS],
        "measured": usable,
        "not_evaluated": missing,
        "per_cell": per_cell,
        "median_span": None,
        "median_half_width": None,
        "ratio": None,
        "complete": not missing,
    }
    if not usable:
        out["status"] = (
            "N1 판정 안 함 — span과 HARD-BAR의 반폭이 같이 있는 셀이 없습니다 "
            f"(없음: {', '.join(missing)})"
        )
        return out
    median_span = statistics.median(float(per_cell[k]["span"]) for k in usable)
    median_width = statistics.median(float(per_cell[k]["half_width"]) for k in usable)
    ratio = median_span / median_width if median_width else None
    out["median_span"] = median_span
    out["median_half_width"] = median_width
    out["ratio"] = ratio
    sizes = f"span 중앙값 {median_span:.4f} · HARD-BAR 반폭 중앙값 {median_width:.4f}"
    if ratio is not None:
        sizes += f" · 비 {ratio:.2f}배"
    if missing:
        out["status"] = (
            f"N1 부분 판정 — 기준은 셀 {len(CELLS)}개에 대한 것인데 {len(usable)}개만 쟀습니다 "
            f"(없음: {', '.join(missing)}). 지금까지 {sizes}"
        )
    elif ratio is not None and ratio > threshold:
        out["status"] = (
            f"N1 충족 — {sizes}. HARD-BAR.md의 주 Δ는 한 실행의 값이 아니라 이 크기의 항이 "
            "섞인 값으로 다시 읽어야 합니다"
        )
    else:
        out["status"] = (
            f"N1 미충족 — {sizes}, 반폭의 {threshold}배 이하입니다. 같은 설정 반복의 산포는 "
            "HARD-BAR.md의 Δ를 다시 읽게 할 만큼 크지 않습니다"
        )
    return out


def n2(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """N2 판정. ``docs/BAR-NOISE.md``에서 인용:

    > HARD-BAR가 **짐**으로 판정한 두 셀(adult 시드43, spambase 시드44)에서, 반복 3회 각각의
    > 주 Δ(``bnoise<r> − bar65nollm``)를 낸다. 세 Δ 다 0 미만이고 세 CI 다 0 아래면 그 짐은
    > **재현된다**고 적는다. 세 Δ 중 하나라도 0 이상이면 **재현되지 않는다**고 적는다. 부호는
    > 셋 다 같지만 CI가 0을 걸치는 반복이 있으면 **약하게 재현**으로 적는다.

    과반이 아니라 셋 중 셋이고 양방향 다 그렇다. HARD-BAR 자신의 기준이 이김에 요구한 기준이
    그것이었고(시드 셋 전원 일치), 짐을 이김보다 느슨한 기준으로 판정해서는 안 되기 때문이다. 예측이
    baseline과 동일하게 나온 짝은 아무것도 재현하지 않은 것으로 센다: 팔이 다르지 않았으니 재현할 Δ가
    없고, 어느 쪽으로든 접으면 0에서 부호를 읽는 것이 된다.
    """
    per_cell: dict[str, dict[str, Any]] = {}
    for dataset, seed in LOST_CELLS:
        key = cell_key(dataset, seed)
        result = results.get((dataset, seed))
        if result is None or result.verdict != "ok":
            per_cell[key] = {
                "verdict": "판정 안 함",
                "reason": "셀이 없거나 판정이 거부됐습니다",
                "deltas": {},
            }
            continue
        pairs = primary_deltas(result)
        deltas = {p.a: p.delta for p in pairs}
        rows = {
            p.a: {
                "delta": p.delta,
                "ci_low": p.ci_low,
                "ci_high": p.ci_high,
                "identical_predictions": p.identical_predictions,
            }
            for p in pairs
        }
        # 판정 안 함이 되는 길 둘을 갈라 둔다. "실행이 다 여기 없다"와 "실행은 다 있는데 그중
        # 하나가 기준선과 정확히 같았다"는 같은 칸에 대한 서로 다른 사실이고, 둘을 한 문자열로 적은
        # 판정 파일은 어느 쪽이 일어났는지 읽는 사람에게 말해 줄 수 없다.
        reason: str | None = None
        if len(pairs) < len(REPEATS):
            verdict = "판정 안 함"
            reason = f"주 Δ가 {len(pairs)}개뿐입니다 — 셋이 다 있어야 세 개를 셀 수 있습니다"
        elif any(p.identical_predictions for p in pairs):
            verdict = "판정 안 함"
            reason = "어느 반복의 예측이 baseline과 동일합니다 — 재현할 Δ가 없습니다"
        elif any(p.delta >= 0 for p in pairs):
            verdict = "재현되지 않음"
        elif all(p.ci_high < 0 for p in pairs):
            verdict = "재현됨"
        else:
            verdict = "약하게 재현"
        per_cell[key] = {"verdict": verdict, "reason": reason, "deltas": deltas, "pairs": rows}
    verdicts = {key: row["verdict"] for key, row in per_cell.items()}
    unjudged = [key for key, v in verdicts.items() if v == "판정 안 함"]
    listing = ", ".join(f"{key} {v}" for key, v in verdicts.items())
    if unjudged and len(unjudged) == len(verdicts):
        status = f"N2 판정 안 함 — 짐 두 셀의 반복이 없습니다 ({', '.join(unjudged)})"
    elif unjudged:
        status = f"N2 부분 — {listing}"
    else:
        status = f"N2 — {listing}"
    return {
        "cells": [cell_key(d, s) for d, s in LOST_CELLS],
        "per_cell": per_cell,
        "verdicts": verdicts,
        "complete": not unjudged,
        "status": status,
    }


def n3(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """N3 판정. ``docs/BAR-NOISE.md``에서 인용:

    > ``bnoisenollm1``·``bnoisenollm2``와 기록된 ``bar65nollm-adult-seed43``의 test 점수 세 점이
    > ``SCORE_TOLERANCE``(1e-6) 안에서 같으면 규칙 팔은 이 조건에서 결정적이라고 적고, 주 Δ의
    > 실행 간 항을 전부 ``llm`` 팔에 귀속한다. 넘으면 **그것이 더 큰 발견**이며, N1·N2의 해석에
    > 그 항을 붙인다.

    기록된 baseline은 비교 대상인 네 번째 것이 아니라 세 점 중 하나다: 여기 모든 주 Δ가 빼는 실행이고,
    그래서 N3가 묻는 것은 새 실행 둘이 서로 맞는가가 아니라 *그 특정한 수*가 재현되는가다.

    0이 아니라 ``SCORE_TOLERANCE``인 이유는 ``bench/paired.py``가 적은 것과 같다: ``logreg``의 예측은
    합의 순서가 스레드 수에 달린 BLAS matmul이고, 그 경로에서 측정된 표류가 5.7e-08이다. 규칙 팔이
    1e-6까지 맞는 것은 "이 실행은 결정적이다"라는 발견이고, float 마지막 비트에 대한 주장이 아니다.
    """
    dataset, seed = RULES_CELL
    key = cell_key(dataset, seed)
    result = results.get((dataset, seed))
    names = [BASELINE, *(rules_arm(r) for r in RULES_REPEATS)]
    scores: dict[str, float] = {}
    if result is not None and result.verdict == "ok":
        for name in names:
            value = (result.arms.get(name) or {}).get("test_reproduced")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[name] = float(value)
    missing = [name for name in names if name not in scores]
    value = (max(scores.values()) - min(scores.values())) if len(scores) > 1 else None
    out: dict[str, Any] = {
        "cell": key,
        "tolerance": SCORE_TOLERANCE,
        "expected_arms": names,
        "scores": scores,
        "not_evaluated": missing,
        "span": value,
        "deterministic": None,
        "complete": not missing,
    }
    if len(scores) < len(names):
        out["status"] = (
            f"N3 판정 안 함 — {key}의 규칙 팔 점수가 {len(scores)}개뿐입니다 "
            f"(없음: {', '.join(missing)}). 주 Δ의 실행 간 항을 llm 팔에 귀속할 근거가 없습니다"
        )
        return out
    assert value is not None
    out["deterministic"] = value <= SCORE_TOLERANCE
    if out["deterministic"]:
        out["status"] = (
            f"N3 확인 — {key}에서 규칙 팔 3점의 span이 {value:.2e}로 허용오차 "
            f"{SCORE_TOLERANCE:g} 안입니다. 주 Δ의 실행 간 항은 전부 llm 팔의 것입니다"
        )
    else:
        out["status"] = (
            f"N3 깨짐 — {key}에서 규칙 팔 3점의 span이 {value:.6f}로 허용오차 "
            f"{SCORE_TOLERANCE:g}를 넘습니다. 이것이 더 큰 발견이고, N1·N2의 해석에 이 항을 "
            "붙여야 합니다"
        )
    return out


def n4(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """N4 판정. ``docs/BAR-NOISE.md``에서 인용:

    > 이 조건의 span 중앙값을 ``REPEATS.md``의 0.0122와 **같은 표에 나란히** 적는다. 판정으로
    > 쓰지 않는다 — 두 측정은 카드 집합도 시드도 바도 코드도 다르고, 세 점의 span은 산포의
    > 추정이 아니다.

    ``observation_only``는 이 docstring의 문장만이 아니라 출력의 필드다. 나중에 JSON을 읽는 사람이
    문서를 안 읽었다는 이유로 이 비교를 판정으로 승격시킬 수 없게 하려는 것이다. 비를 계산하지 않는
    이유도 같다: 두 수를 나누는 것이 그중 하나를 바로 만드는 행위이고, 이건 바가 아니다.
    """
    spans = {
        cell_key(d, s): span(results[(d, s)])
        for d, s in CELLS
        if (d, s) in results and span(results[(d, s)]) is not None
    }
    values = [float(v) for v in spans.values() if v is not None]
    median = statistics.median(values) if values else None
    if median is None:
        status = "N4 관찰 없음 — span을 낼 수 있는 셀이 없습니다"
    else:
        relation = "작습니다" if median < REPEATS_NOISE_FLOOR else "작지 않습니다"
        status = (
            f"N4 관찰 — 셀 {len(values)}개의 span 중앙값 {median:.4f}, REPEATS.md의 "
            f"{REPEATS_NOISE_FLOOR}보다 {relation}. 판정으로 쓰지 않습니다"
        )
    return {
        "observation_only": True,
        "repeats_noise_floor": REPEATS_NOISE_FLOOR,
        "repeats_noise_floor_source": REPEATS_NOISE_FLOOR_SOURCE,
        "median_span": median,
        "per_cell": spans,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #


def report_spans(
    results: dict[tuple[str, int], DatasetVerdict], half_widths: dict[str, float]
) -> None:
    print()
    print("반복 3회의 test 점수 span — HARD-BAR의 주 Δ 반폭 옆에서")
    print(f"   {'셀':<24} {'span':>9} {'HARD-BAR 반폭':>14} {'비':>7}  {'스레드':<9} {'모양':<7}")
    for dataset, seed in CELLS:
        key = cell_key(dataset, seed)
        result = results.get((dataset, seed))
        if result is None:
            print(f"   {key:<24} {'실행 없음':>9}")
            continue
        value = span(result)
        width = half_widths.get(key)
        pins = {arm.get("threads_pinned") for arm in result.arms.values()}
        if result.verdict != "ok":
            threads = "판정 거부"
        elif pins == {True}:
            threads = "1로 고정"
        elif pins <= {None}:
            threads = "기록 없음"
        else:
            threads = "고정 아님"
        agree = shape_agreement(result)["agree"]
        shape = "" if agree is None else ("같음" if agree else "다름")
        cells = (
            f"{value:>9.4f}" if value is not None else f"{'—':>9}",
            f"{width:>14.4f}" if width is not None else f"{'—':>14}",
            f"{value / width:>7.2f}" if (value is not None and width) else f"{'—':>7}",
        )
        print(f"   {key:<24} {' '.join(cells)}  {threads:<9} {shape:<7}")


def payload(
    results: dict[tuple[str, int], DatasetVerdict],
    half_widths: dict[str, float],
    resamples: int,
) -> dict[str, Any]:
    ordered = [cell for cell in CELLS if cell in results]
    return {
        "generated_by": "bench/bar_noise.py",
        "preregistration": "docs/BAR-NOISE.md",
        "measures_the_ruler_of": "docs/HARD-BAR.md",
        "margin": BAR_MARGIN,
        "repeats": list(REPEATS),
        "baseline": BASELINE,
        "run_seeds": sorted({seed for _, seed in ordered}),
        # 부트스트랩 시드. 파일 전체에 하나다: ``bench/recheck.py``가 판정마다 ``seed`` 하나를
        # 읽고 그걸로 모든 짝을 다시 계산한다.
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # 실행들이 쓰는 것과 같은 기록기다. 그래서 이 판정과 그 아래 시도들이 같은 환경을 적는다 —
        # 환경 변수만으로는 정해지지 않는 ``cpu_count``까지.
        "threads": thread_state(),
        # 이름이 하는 일 그대로다: 이 파일은 HARD-BAR의 판정을 다시 계산하지 않고 거기서 폭만
        # 읽는다.
        "hard_bar_half_widths": half_widths,
        "datasets": [
            {
                "dataset": results[cell].dataset,
                "run_seed": cell[1],
                "metric": results[cell].metric,
                "task": results[cell].task,
                "verdict": results[cell].verdict,
                "refusal": results[cell].refusal,
                "test_rows": results[cell].test_rows,
                "test_fingerprint": results[cell].test_fingerprint,
                "missing_arms": results[cell].missing_arms,
                # 스레드 상태를 확인할 수 없었던 실행들. 팔별 블록 옆에 따로 저장하지 않고
                # 거기서 유도하므로 둘이 어긋날 수 없다.
                "threads_unrecorded": [
                    name for name, arm in results[cell].arms.items() if arm.get("threads") is None
                ],
                "span": span(results[cell]),
                "shape_agreement": shape_agreement(results[cell]),
                "arms": results[cell].arms,
                "pairs": [vars(p) for p in results[cell].pairs],
            }
            for cell in ordered
        ],
        "bar_check": bar_check(results),
        "N1": n1(results, half_widths),
        "N2": n2(results),
        "N3": n3(results),
        "N4": n4(results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "올린 바에서 같은 설정 반복이 얼마나 흔들리는가 (docs/BAR-NOISE.md의 사전 등록 기준)"
        )
    )
    parser.add_argument(
        "names",
        nargs="*",
        help=f"데이터셋 이름 (생략하면 {' '.join(d for d, _ in CELLS)})",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/bar-noise.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scope = {dataset for dataset, _ in CELLS}
    unknown = [name for name in args.names if name not in scope]
    if unknown:
        print(
            f"사전 등록의 범위 밖: {', '.join(unknown)} (셀 {', '.join(sorted(scope))})",
            file=sys.stderr,
        )
        return 2
    wanted = set(args.names) or scope

    print(
        f"잡음 바닥 판정 · margin {BAR_MARGIN} · 반복 {len(REPEATS)}회 · "
        f"재추출 {args.resamples}회 (재추출 시드 {RESAMPLE_SEED}) · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: dict[tuple[str, int], DatasetVerdict] = {}
    for dataset, seed in CELLS:
        if dataset not in wanted:
            continue
        collected: dict[str, Scored] = {}
        results[(dataset, seed)] = adjudicate(dataset, seed, args.resamples, collected)
        if collected:
            legs[results[(dataset, seed)].dataset] = collected
    report(list(results.values()))

    half_widths = recorded_half_widths()
    if not half_widths:
        print()
        print(
            f"HARD-BAR의 반폭을 읽을 수 없습니다 ({HARD_BAR_VERDICT.as_posix()}) — "
            "N1은 판정하지 않습니다",
            file=sys.stderr,
        )
    report_spans(results, half_widths)

    out_path = args.out or OUT_DIR / "bar-noise.json"
    if not legs:
        # 채점된 것이 없으니 공표할 것도, 공표할 번들도 없다. 여기서 판정 파일을 쓰면 인수 없는
        # ``python -m bench.recheck``이 읽는 디렉터리에 놓여서 그걸 실패시킨다.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_bar_noise.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, half_widths, args.resamples), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results.values()})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")

    print()
    print(f"바 — HARD-BAR의 예측 대조 (허용 {BAR_TOLERANCE:g}):")
    for row in bar_check(results):
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} {row['arm']:<14} "
            f"예측={row['predicted_bar']:.4f} 실측={row['actual_bar']} {mark}"
        )
    print()
    for statement in (
        n1(results, half_widths),
        n2(results),
        n3(results),
        n4(results),
    ):
        print(statement["status"])
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results.values() if r.verdict != "ok"]
    if refused:
        print(f"거부된 셀: {', '.join(refused)}", file=sys.stderr)
        return 1
    # 무료 절반만 돌린 실행이 여기 온다: 건드린 셀은 다 내부적으로 일관되지만 사전 등록된 팔이
    # 아직 다 들어오지 않았다. exit 0은 "진술 넷에 답했다"로 읽히고, 그중 둘은 아니다.
    incomplete = {r.dataset: r.missing_arms for r in results.values() if r.missing_arms}
    if incomplete:
        listing = "; ".join(f"{key}: {', '.join(arms)}" for key, arms in incomplete.items())
        print(f"아직 돌지 않은 팔이 있습니다 — {listing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
