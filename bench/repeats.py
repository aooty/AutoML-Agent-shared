"""``docs/REPEATS.md``의 판정: 아무것도 움직이지 않을 때 ``llm`` 팔은 얼마나 움직이는가.

사전 등록은 ``docs/REPEATS.md``이고, **이 파일은 그 실행들이 존재하기 전에 커밋됐다** — 다른 판정
모듈 둘이 따르는 것과 같은 규칙, 같은 이유다. 숫자를 보고 쓴 스크립트는 그 숫자에 맞출 수 있고,
쓸 때 아무리 조심했다고 해도 그것을 증명하지 못한다. 증명은 git 순서다.

축 둘. 서로 다른 질문에 답하므로 따로 둔다. **R**은 설정 하나를 세 번 반복하므로 그 Δ가 담는 것은
응답자가 동일한 프롬프트에 세 번 답한 것뿐이다. **S**는 시드를 옮겨 분할을 옮기므로 그 산포는 그
위에 분할까지 담는다. 함께 평균 내면 둘 중 아무것도 말하지 않게 된다.

R은 부트스트랩할 수 있고 S는 할 수 없는 이유. R의 세 실행은 카드와 시드를 공유하므로 test 행도
공유한다 — 둘의 차이는 그 행들에 대한 짝지은 차이이고, 구간은 ``paired.py``가 내는 그것이다. S의
실행들은 각자 자기 행을 나누므로 짝지을 것이 없고, 그래서 S는 점추정 세 개의 폭을 보고하며 관찰로
표시된다. ``REPEATS.md``가 그 비대칭을 사전 등록했다(S1: "시드 3개는 산포의 추정이 아니라 세
점입니다").

기록된 ``RESULTS.md`` 실행을 빼지 않는 이유. ``main``은 그 실행들보다 행동을 바꾸는 커밋 여러
개만큼 앞에 있다(xgboost early stopping, ``internal_validation``, row budget 프롬프트 절, 스레드
기록). 그래서 그런 Δ는 코드 차이와, 이 실험이 떼어 내려는 실행 간 항을 *둘 다* 담는다.
``SPG.md``의 관찰 1이 그 규칙을 깬 Δ이고, 그래서 판정이 아니라 관찰로 냈으며 여기서 제대로 잰다.

스레드 상태는 가정하지 않고 확인한다. A2는 스레드 수에 따라 balanced_accuracy가 0.0077 벌어지는
것을 쟀고, 그건 짝지은 판정 하나의 반폭의 0.99배 — R이 재려는 것과 같은 크기다. 이제 모든 실행이
자기 스레드 상태를 기록하므로, 이 스크립트는 그것을 되읽고 반복들이 같은 상태에서 돌지 않은
데이터셋을 거부한다. *기록이 없는* 상태는 거부가 아니라 유보다: "확인하지 않았다"와 "확인했고
다르다"는 다른 사실이다(:func:`automl_agent.threads.thread_state_changed`).

팔들이 돌았던 것처럼 ``OMP_NUM_THREADS=1``로 돌린다. 값은 출력에 기록된다.

사용법:
    python -m bench.repeats                     # 두 축, 모든 데이터셋
    python -m bench.repeats spambase adult
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import THREAD_ENV, describe_thread_state, thread_state, thread_state_changed
from bench.datasets import ALL, BY_NAME, CHEAP_SEEDS, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
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

# R 축: 명령줄이 ``--thread-id``만 다른 세 실행. 그 값은 체크포인트 키이고 어떤 프롬프트에도
# 닿지 않는다. 이름 대신 번호인 이유는 이름 붙일 것이 없기 때문이다 — 여기서는 어느 팔도
# 대조군이 아니다.
REPEATS: tuple[int, ...] = (1, 2, 3)
# S 축. 리터럴이 아니라 ``CHEAP_SEEDS``인 이유: 카드가 이미 커밋돼 있고 ``no_llm``/``random``
# 수가 이미 ``RESULTS.md``에 있는 시드들이라, 애초에 S의 폭을 그 문서의 시드 산포와 견줄 수 있게
# 하는 것이 그것이다.
SEEDS: tuple[int, ...] = CHEAP_SEEDS
# S의 시드 42 점은 네 번째 실행이 아니라 반복 1에서 온다. 그 시드에서는 R의 세 반복이 똑같이
# 유효한 점이고, 그중 S의 폭을 가장 넓게(또는 좁게) 만드는 것을 고르면 숫자를 보고 고르는 것이 된다.
S_REPEAT = REPEATS[0]

# R1의 문턱, 사전 등록에서 인용: span 중앙값이 반폭 중앙값의 이 배수를 넘으면, 공개된 모든 Δ를
# 이 크기의 항이 섞인 값으로 다시 읽어야 한다.
R1_THRESHOLD = 0.5
# S1을 나란히 적으라고 사전 등록된 수: ``RESULTS.md``에서 random 팔의 시드 폭 평균. 다시 계산하지
# 않고 상수로 두는 이유는 그것이 *공개된* 수치이고, 여기 두는 목적이 읽는 사람이 이 줄을 그 문서와
# 맞춰 볼 수 있게 하는 것이기 때문이다.
RANDOM_SEED_SPAN = 0.1472


def run_dir(dataset: Dataset, repeat: int, seed: int) -> Path:
    """``bench/scripts/run_repeats.sh``가 실행 하나를 둔 곳. thread-id가 디렉터리 이름이다."""
    return ARTIFACTS_DIR / f"rep{repeat}-{dataset.name}-seed{seed}"


# --------------------------------------------------------------------------- #
# 각 실행이 실제로 돌았던 환경
# --------------------------------------------------------------------------- #


def recorded_threads(winner: Winner) -> dict[str, Any] | None:
    """우승한 시도의 스레드 상태. 그 시도 자신의 ``result.json``에서 읽는다.

    ``run_config.json``이 아니라 시도의 사본을 읽는다 — resume은 실행 수준 파일을 덮어쓰므로, 그
    파일은 수를 견주고 있는 그 적합이 아니라 실행의 가장 최근 구간을 서술한다. 이 실행들이 resume될
    일은 없지만, 아무 일도 잘못되지 않았을 때만 맞는 확인은 확인이 아니다.
    """
    result = read_json_object(winner.directory / "result.json")
    if result is None:
        return None
    threads = result.get("threads")
    return threads if isinstance(threads, dict) else None


def pinned(threads: dict[str, Any] | None) -> bool | None:
    """변수 셋이 모두 1로 고정됐는가. 기록이 없으면 ``None``.

    고정은 ``run_repeats.sh``가 하는 일이고 ``REPEATS.md``가 했다고 적은 것이다. 이 함수는 그것을
    산출물에서 되읽는다. 그 자체로 거부 사유는 아니다 — 일관되게 고정 아닌 상태도 응답자의 산포는
    잰다 — 그래도 출력에 들어가야 한다. 고정 안 된 실행의 산포에는 스레드 항이 접혀 들어가 있어서
    R의 답으로 인용할 수 없기 때문이다.
    """
    if threads is None:
        return None
    if not all(key in threads for key in THREAD_ENV):
        return None
    return all(str(threads.get(key)) == "1" for key in THREAD_ENV)


def check_one_environment(threads: dict[str, dict[str, Any] | None], label: str) -> list[str]:
    """두 실행이 서로 다른 스레드 상태에서 적합된 것이 확인되면 거부한다.

    돌려주는 것은 상태가 *기록되지 않은* 이름들이고, 그건 거부가 아니라 유보다. 하네스가 그 필드를
    기록하기 전에 만든 실행은 확인할 수 없고, 그렇다고 말하는 것은 일치했다고 말하는 것과 다르다.
    """
    names = sorted(threads)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if thread_state_changed(threads[a], threads[b]):
                raise Refusal(
                    f"{label}: {a}와 {b}가 다른 스레드 상태에서 적합됐습니다 "
                    f"({describe_thread_state(threads[a])} vs {describe_thread_state(threads[b])}) — "
                    "A2가 잰 스레드 span 0.0077은 이 실험이 재려는 것과 같은 크기입니다"
                )
    return [name for name in names if threads[name] is None]


# --------------------------------------------------------------------------- #
# R — 설정 하나의 세 실행
# --------------------------------------------------------------------------- #


def adjudicate_repeats(
    dataset: Dataset,
    resamples: int,
    seed: int = JUDGED_SEED,
    legs: dict[str, Scored] | None = None,
) -> DatasetVerdict:
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate_repeats(dataset, resamples, seed, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


def _adjudicate_repeats(
    dataset: Dataset,
    resamples: int,
    seed: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    metric = dataset.metric
    winners: dict[str, Winner] = {}
    for repeat in REPEATS:
        arm = f"llm_r{repeat}"
        winner = loop_winner(arm, run_dir(dataset, repeat, seed), metric)
        if winner is None:
            out.missing_arms.append(arm)
        else:
            winners[arm] = winner
    if out.missing_arms:
        # 셋 다이거나 아무것도 아니다. 세 점 중 둘에 대한 폭은 R1이 말하는 양이 아니고, 그것인
        # 척 보고하면 구조적으로 산포를 작게 말하는 것이 된다.
        raise Refusal(
            f"{dataset.name}: 반복 {', '.join(out.missing_arms)}이 없습니다 — "
            f"R은 {len(REPEATS)}회가 다 있어야 span이 사전 등록된 그 span입니다"
        )

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
            f"{dataset.name}: 반복마다 test 지문이 다릅니다 ({listing}) — "
            "같은 카드·같은 시드인데 행이 움직였다면 재는 대상이 응답 차이가 아닙니다"
        )
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset.name}: 분할을 정하는 config 필드가 반복마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )
    # 두 반복이 다른 스레드 상태에서 적합된 것이 확인되면 예외를 낸다. 돌려주는 기록 없는 이름들에
    # 따로 필드를 둘 필요는 없다 — 아래에서 ``threads``를 팔마다 쓰고, 거기 없다는 것이 같은 말을
    # 하므로 발을 맞춰야 할 두 번째 자리를 만들지 않는다.
    check_one_environment(threads, dataset.name)

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
            # 데이터셋이 아니라 실행마다 둔다. 이 필드의 요점 자체가 두 실행이 여기서 다를 수
            # 있다는 것이고, 데이터셋 수준 사본은 그중 하나만 적을 수 있다.
            "threads": threads[name],
            "threads_pinned": pinned(threads[name]),
        }

    # 순서 없는 모든 쌍. 작은 반복을 앞에 두어 부호를 모든 줄에서 같게 읽는다.
    names = sorted(scored)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out.pairs.append(paired_delta(scored[a], scored[b], metric, resamples, seed, "repeat"))

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다 — ``bench/paired.py``와 같은 규칙.
    if legs is not None:
        legs.update(scored)


def span(result: DatasetVerdict) -> float | None:
    """반복들이 다시 유도한 test 점수의 최대 − 최소. 낼 수 없으면 ``None``."""
    if result.verdict != "ok":
        return None
    scores = [
        float(arm["test_reproduced"])
        for arm in result.arms.values()
        if isinstance(arm.get("test_reproduced"), (int, float))
    ]
    if len(scores) < len(REPEATS):
        return None
    return max(scores) - min(scores)


# --------------------------------------------------------------------------- #
# R1이 비교 대상으로 삼는 반폭
# --------------------------------------------------------------------------- #


def recorded_half_widths(seed: int = JUDGED_SEED, path: Path | None = None) -> dict[str, float]:
    """데이터셋마다, *공개된* 판정의 주 비교들이 낸 반폭의 중앙값.

    다시 계산하지 않고 ``bench/runs/paired/seed42.json``에서 읽는다. R1이 말하는 양은
    ``RESULTS.md``가 실제로 낸 구간의 폭이고, 그 문서를 읽는 사람에게 다시 읽으라고 하는 것이
    그것이기 때문이다.

    주 쌍만 본다 — 기준이 그것으로 정해진, 예산을 맞춘 비교들이다. 그것들의 중앙값을 쓰는 이유는
    반폭이 어느 쌍에 속하는지보다 test 행과 지표의 성질에 훨씬 더 가깝고, 중앙값은 어느 쌍이 우연히
    가장 넓게 재추출됐는지 신경 쓰지 않기 때문이다.
    """
    verdict = read_json_object(path or OUT_DIR / f"seed{seed}.json")
    if verdict is None:
        return {}
    widths: dict[str, float] = {}
    for entry in verdict.get("datasets") or []:
        halves = [
            (float(p["ci_high"]) - float(p["ci_low"])) / 2
            for p in entry.get("pairs") or []
            if p.get("kind") == "primary"
            and isinstance(p.get("ci_low"), (int, float))
            and isinstance(p.get("ci_high"), (int, float))
        ]
        if halves:
            widths[str(entry.get("dataset"))] = statistics.median(halves)
    return widths


# --------------------------------------------------------------------------- #
# S — 시드 레버
# --------------------------------------------------------------------------- #


@dataclass
class SeedRow:
    """데이터셋 하나의 시드별 ``llm`` test 점수와, 그것들의 폭."""

    dataset: str
    metric: str
    task: str
    verdict: str = "ok"
    refusal: str | None = None
    scores: dict[int, float] = field(default_factory=dict)
    fingerprints: dict[int, str] = field(default_factory=dict)
    missing_seeds: list[int] = field(default_factory=list)
    caveat: str | None = None

    @property
    def span(self) -> float | None:
        if self.verdict != "ok" or len(self.scores) < 2:
            return None
        return max(self.scores.values()) - min(self.scores.values())


def seed_span(dataset: Dataset) -> SeedRow:
    out = SeedRow(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _seed_span(dataset, out)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
    return out


def _seed_span(dataset: Dataset, out: SeedRow) -> None:
    threads: dict[str, dict[str, Any] | None] = {}
    for seed in SEEDS:
        winner = loop_winner(f"llm_s{seed}", run_dir(dataset, S_REPEAT, seed), dataset.metric)
        if winner is None:
            out.missing_seeds.append(seed)
            continue
        result = reproduce(winner, dataset.metric)
        gap = abs(result.test_score - winner.recorded_test)
        if gap > SCORE_TOLERANCE:
            raise Refusal(
                f"시드 {seed}: 다시 유도한 test 점수 {result.test_score:.6f}가 기록된 "
                f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
            )
        out.scores[seed] = result.test_score
        out.fingerprints[seed] = result.fingerprint
        threads[f"시드 {seed}"] = recorded_threads(winner)
    if len(out.scores) < 2:
        raise Refusal(
            f"{dataset.name}: 시드가 {len(out.scores)}개뿐입니다 — 폭을 낼 수 없습니다"
        )
    if len(set(out.fingerprints.values())) < len(out.fingerprints):
        # 시드는 둘인데 test 행은 한 벌이다: 시드가 분할에 닿지 않았으므로 이 "시드 폭"은 이름을
        # 잘못 달고 있는 반복 폭이다. 결과가 아니라 실행의 버그다.
        listing = ", ".join(f"{seed}={fp[:12]}" for seed, fp in sorted(out.fingerprints.items()))
        raise Refusal(
            f"{dataset.name}: 시드가 다른데 test 지문이 겹칩니다 ({listing}) — "
            "카드의 protocol.seed가 --seed와 함께 움직이지 않았습니다"
        )
    unrecorded = check_one_environment(threads, dataset.name)
    if unrecorded:
        out.caveat = (
            f"{', '.join(unrecorded)}의 스레드 상태가 기록에 없습니다 — "
            "고정이 지켜졌는지 산출물로는 확인할 수 없습니다"
        )


# --------------------------------------------------------------------------- #
# 사전 등록된 진술
# --------------------------------------------------------------------------- #


def _partial(expected: set[str], measured: list[str]) -> list[str]:
    return sorted(expected - set(measured))


def r1(
    results: list[DatasetVerdict], half_widths: dict[str, float], threshold: float = R1_THRESHOLD
) -> dict[str, Any]:
    """R1 판정. ``docs/REPEATS.md``에서 인용:

    > 이진 4개에서 `llm` 팔 반복 3회의 test 점수 span의 **중앙값**을, 같은 데이터셋에서 한 번
    > 측정의 짝지은 판정 **반폭 중앙값**과 비교한다. span 중앙값이 반폭 중앙값의 **0.5배를
    > 넘으면**, `RESULTS.md`와 `SPG.md`가 낸 모든 Δ는 "한 실행의 값"이 아니라 "이 크기의 항이
    > 섞인 값"으로 다시 읽어야 한다고 적는다.

    두 중앙값은 *같은* 데이터셋에 대해 낸다 — span과 공개된 반폭이 둘 다 있는 것들이다. 서로 다른
    집합에서 내면 쉬운 데이터셋의 중앙값과 넓은 데이터셋의 중앙값을 견주고 그 비를 측정이라고 부르는
    것이 된다.

    이진만 본다. 못박은 4개에 못 미치는 범위는 답이 아니라 부분 판정이라고 적는다. R1은 방향에 대해
    아무 말도 하지 않는다 — 작은 span은 작은 span으로 보고되고, 그게 이 저장소에 유리한 결과인데,
    그것을 내는 문장이 같은 문장이다.
    """
    expected = set(JUDGED_NAMES)
    binary = [r for r in results if r.dataset in expected]
    per_dataset: dict[str, dict[str, Any]] = {}
    for result in binary:
        width = half_widths.get(result.dataset)
        value = span(result)
        per_dataset[result.dataset] = {
            "span": value,
            "half_width": width,
            "ratio": (value / width) if (value is not None and width) else None,
        }
    usable = sorted(
        name
        for name, row in per_dataset.items()
        if row["span"] is not None and row["half_width"] is not None
    )
    missing = _partial(expected, usable)
    out: dict[str, Any] = {
        "threshold": threshold,
        "expected_binary": sorted(expected),
        "measured": usable,
        "not_evaluated": missing,
        "per_dataset": per_dataset,
        "median_span": None,
        "median_half_width": None,
        "ratio": None,
        "complete": not missing,
    }
    if not usable:
        out["status"] = (
            "R1 판정 안 함 — span과 공개된 반폭이 같이 있는 이진 데이터셋이 없습니다 "
            f"(없음: {', '.join(missing)})"
        )
        return out
    median_span = statistics.median(float(per_dataset[n]["span"]) for n in usable)
    median_width = statistics.median(float(per_dataset[n]["half_width"]) for n in usable)
    ratio = median_span / median_width if median_width else None
    out["median_span"] = median_span
    out["median_half_width"] = median_width
    out["ratio"] = ratio
    sizes = f"span 중앙값 {median_span:.4f} · 반폭 중앙값 {median_width:.4f}"
    if ratio is not None:
        sizes += f" · 비 {ratio:.2f}배"
    if missing:
        out["status"] = (
            f"R1 부분 판정 — 기준은 이진 {len(expected)}개에 대한 것인데 {len(usable)}개만 "
            f"쟀습니다 (없음: {', '.join(missing)}). 지금까지 {sizes}"
        )
    elif ratio is not None and ratio > threshold:
        out["status"] = (
            f"R1 충족 — {sizes}. RESULTS.md와 SPG.md의 모든 Δ는 이 크기의 항이 섞인 값으로 "
            "다시 읽어야 합니다"
        )
    else:
        out["status"] = (
            f"R1 미충족 — {sizes}, 반폭의 {threshold}배 이하입니다. 같은 설정 반복의 산포는 "
            "그 문서들의 Δ를 다시 읽게 할 만큼 크지 않습니다"
        )
    return out


def r2(results: list[DatasetVerdict]) -> dict[str, Any]:
    """R2 판정. ``docs/REPEATS.md``에서 인용:

    > 반복 3회를 짝지어 낸 Δ 3쌍 중 **95% CI가 0을 벗어나는 쌍이 하나라도 있으면**, 같은 설정
    > 반복이 짝지은 판정에서 유의한 차이를 낼 수 있다는 것이 확인된 것으로 적는다.

    이진 집합 어디든 쌍 하나면 충분하다. 주장이 존재 주장이기 때문이다 — 이 벤치마크가 내내 쓴 자가
    *하나의* 설정으로 돈 두 실행을 다르다고 부를 수 있다는 것. 예측이 동일하게 나온 쌍은 어느 쪽의
    증거도 아니라 따로 센다. 그건 같은 설정이 정확히 재현된 것이고, 반대 방향의 발견이라 자기 수를
    가질 만하다.
    """
    expected = set(JUDGED_NAMES)
    excluding_zero: list[dict[str, Any]] = []
    crossing = 0
    identical = 0
    measured: list[str] = []
    for result in results:
        if result.dataset not in expected:
            continue
        if result.verdict != "ok" or not result.pairs:
            continue
        measured.append(result.dataset)
        for pair in result.pairs:
            if pair.identical_predictions:
                identical += 1
            elif pair.ci_low > 0 or pair.ci_high < 0:
                excluding_zero.append(
                    {
                        "dataset": result.dataset,
                        "pair": f"{pair.a} − {pair.b}",
                        "delta": pair.delta,
                        "ci_low": pair.ci_low,
                        "ci_high": pair.ci_high,
                    }
                )
            else:
                crossing += 1
    missing = _partial(expected, measured)
    total = len(excluding_zero) + crossing + identical
    counts = f"0을 벗어남 {len(excluding_zero)} / 걸침 {crossing} / 예측 동일 {identical}"
    if not measured:
        status = f"R2 판정 안 함 — 판정된 이진 데이터셋이 없습니다 (없음: {', '.join(missing)})"
    elif excluding_zero:
        where = ", ".join(f"{row['dataset']} {row['pair']}" for row in excluding_zero)
        status = (
            f"R2 확인 — 같은 설정 반복이 유의한 짝지은 차이를 냈습니다 ({where}). {counts}"
        )
    elif missing:
        status = (
            f"R2 부분 — {len(measured)}개만 쟀고 0을 벗어난 쌍은 없습니다 "
            f"(없음: {', '.join(missing)}). {counts}"
        )
    else:
        status = f"R2 미확인 — 이 데이터와 이 재추출 수로는 구분되지 않습니다. {counts}"
    return {
        "measured": sorted(measured),
        "not_evaluated": missing,
        "pairs_total": total,
        "pairs_excluding_zero": excluding_zero,
        "pairs_crossing_zero": crossing,
        "pairs_identical": identical,
        "complete": not missing,
        "status": status,
    }


def s1(rows: list[SeedRow], random_span: float = RANDOM_SEED_SPAN) -> dict[str, Any]:
    """S1 판정. ``docs/REPEATS.md``에서 인용:

    > 시드 42·43·44의 `llm` 팔 test 점수 span을 데이터셋마다 내고, `RESULTS.md`의 `random` 팔
    > 시드 폭(평균 0.1472)과 **같은 표에 나란히** 적는다. `llm` 팔의 span이 더 작으면 그것은 이
    > 저장소의 주장에 유리한 관찰이므로, **그렇게만 적고 판정으로 쓰지 않는다**.

    ``observation_only``을 이 docstring에만 두지 않고 출력의 필드로 두는 이유는, 나중에 JSON을 읽는
    사람이 문서를 안 읽었다는 이유로 이 비교를 판정으로 승격시킬 수 없게 하는 것이다. 시드 셋은 점
    셋이다. 폭 네 개의 평균을 적는 것은 ``RESULTS.md``가 random 팔에 대해 그렇게 적었기 때문이고,
    종류가 다른 요약끼리 견주는 것은 거친 것보다 나쁘다.
    """
    expected = set(JUDGED_NAMES)
    per_dataset = {
        row.dataset: {
            "scores": {str(seed): value for seed, value in sorted(row.scores.items())},
            "span": row.span,
            "seeds": sorted(row.scores),
            "verdict": row.verdict,
            "refusal": row.refusal,
        }
        for row in rows
    }
    spans = [
        float(row.span)
        for row in rows
        if row.dataset in expected and row.span is not None
    ]
    measured = sorted(
        row.dataset
        for row in rows
        if row.dataset in expected and row.span is not None
    )
    missing = _partial(expected, measured)
    mean_span = statistics.fmean(spans) if spans else None
    if mean_span is None:
        status = f"S1 관찰 없음 — 시드 폭을 낼 수 있는 이진 데이터셋이 없습니다 (없음: {', '.join(missing)})"
    else:
        scope = f"이진 {len(spans)}개" + (f" (없음: {', '.join(missing)})" if missing else "")
        relation = "작습니다" if mean_span < random_span else "작지 않습니다"
        status = (
            f"S1 관찰 — {scope}에서 llm 팔의 시드 폭 평균 {mean_span:.4f}, "
            f"RESULTS.md의 random 팔 {random_span:.4f}보다 {relation}. 판정으로 쓰지 않습니다"
        )
    return {
        "observation_only": True,
        "random_seed_span": random_span,
        "seeds": list(SEEDS),
        "expected_binary": sorted(expected),
        "measured": measured,
        "not_evaluated": missing,
        "mean_span": mean_span,
        "per_dataset": per_dataset,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #


def report_spans(results: list[DatasetVerdict], half_widths: dict[str, float]) -> None:
    print()
    print("R — 같은 설정 반복 3회의 test 점수 span")
    print(f"   {'데이터셋':<16} {'span':>9} {'공개 반폭':>11} {'비':>7}  {'스레드':<9}")
    for result in results:
        value = span(result)
        width = half_widths.get(result.dataset)
        pins = {arm.get("threads_pinned") for arm in result.arms.values()}
        if result.verdict != "ok":
            threads = "판정 거부"
        elif pins == {True}:
            threads = "1로 고정"
        elif pins <= {None}:
            threads = "기록 없음"
        else:
            threads = "고정 아님"
        cells = (
            f"{value:>9.4f}" if value is not None else f"{'—':>9}",
            f"{width:>11.4f}" if width is not None else f"{'—':>11}",
            f"{value / width:>7.2f}" if (value is not None and width) else f"{'—':>7}",
        )
        print(f"   {result.dataset:<16} {' '.join(cells)}  {threads:<9}")


def report_seeds(rows: list[SeedRow]) -> None:
    print()
    print(f"S — 시드 {', '.join(str(s) for s in SEEDS)}의 test 점수 (관찰, 판정 아님)")
    header = "".join(f"{f'시드 {seed}':>10}" for seed in SEEDS)
    print(f"   {'데이터셋':<16}{header} {'폭':>9}")
    for row in rows:
        if row.verdict != "ok":
            print(f"   {row.dataset:<16} 관찰 거부: {row.refusal}")
            continue
        cells = "".join(
            f"{row.scores[seed]:>10.4f}" if seed in row.scores else f"{'—':>10}" for seed in SEEDS
        )
        width = f"{row.span:>9.4f}" if row.span is not None else f"{'—':>9}"
        print(f"   {row.dataset:<16}{cells} {width}")
        if row.caveat:
            print(f"      {row.caveat}")


def payload(
    results: list[DatasetVerdict],
    seed_rows: list[SeedRow],
    half_widths: dict[str, float],
    resamples: int,
) -> dict[str, Any]:
    return {
        "generated_by": "bench/repeats.py",
        "preregistration": "docs/REPEATS.md",
        "seed": JUDGED_SEED,
        "repeats": list(REPEATS),
        "seeds": list(SEEDS),
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # 실행들이 쓰는 것과 같은 기록기라, 이 판정과 그 아래 시도들이 같은 환경을 서술한다 —
        # 변수만으로는 정해지지 않는 ``cpu_count``까지 포함해서.
        "threads": thread_state(),
        # 있는 대로 이름 붙인다: 이 문서는 RESULTS.md의 판정을 다시 계산하지 않고, 거기서 폭만
        # 읽는다.
        "published_half_widths": half_widths,
        "repeat_datasets": [
            {
                "dataset": r.dataset,
                "metric": r.metric,
                "task": r.task,
                "verdict": r.verdict,
                "refusal": r.refusal,
                "test_rows": r.test_rows,
                "test_fingerprint": r.test_fingerprint,
                "missing_arms": r.missing_arms,
                # 스레드 상태를 확인할 수 없었던 반복. 팔별 블록 옆에 따로 저장하지 않고
                # 거기서 유도하므로 둘이 어긋날 수 없다.
                "threads_unrecorded": [
                    name for name, arm in r.arms.items() if arm.get("threads") is None
                ],
                "span": span(r),
                "arms": r.arms,
                "pairs": [vars(p) for p in r.pairs],
            }
            for r in results
        ],
        "R1": r1(results, half_widths),
        "R2": r2(results),
        "S1": s1(seed_rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="llm 팔의 반복·시드 산포 (docs/REPEATS.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help="이름 (기본: 전부)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help=f"기본: bench/runs/paired/repeats-seed{JUDGED_SEED}.json"
    )
    parser.add_argument(
        "--skip-seeds",
        action="store_true",
        help="R만 판정합니다 (S의 실행 10회가 아직 없을 때)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or [d.name for d in ALL]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"반복 판정 · 반복 {len(REPEATS)}회 · 시드 {JUDGED_SEED} · 재추출 {args.resamples}회 · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results = []
    for name in names:
        collected: dict[str, Scored] = {}
        results.append(adjudicate_repeats(BY_NAME[name], args.resamples, legs=collected))
        if collected:
            legs[name] = collected
    report(results)
    half_widths = recorded_half_widths()
    if not half_widths:
        print()
        print(
            f"공개된 반폭을 읽을 수 없습니다 ({(OUT_DIR / f'seed{JUDGED_SEED}.json').as_posix()}) "
            "— R1은 판정하지 않습니다",
            file=sys.stderr,
        )
    report_spans(results, half_widths)

    seed_rows = [] if args.skip_seeds else [seed_span(BY_NAME[name]) for name in names]
    if seed_rows:
        report_seeds(seed_rows)

    out_path = args.out or OUT_DIR / f"repeats-seed{JUDGED_SEED}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, seed_rows, half_widths, args.resamples), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {r.dataset: r.metric for r in results})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print(r1(results, half_widths)["status"])
    print(r2(results)["status"])
    print("S1 건너뜀 (--skip-seeds)" if args.skip_seeds else s1(seed_rows)["status"])
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"R 판정이 거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
