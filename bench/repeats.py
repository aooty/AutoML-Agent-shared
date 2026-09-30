"""Pre-registered check: how much does the ``llm`` arm move when nothing changes?

Roles:

* Settings — repeats, seeds, and pre-registered numbers.
* Thread checks — read and compare each run's thread state.
* R axis — three repeats of one setup, paired.
* Half widths — published half widths that R1 compares against.
* S axis — ``llm`` test scores across seeds.
* Verdicts — R1, R2, and S1 from ``docs/REPEATS.md``.
* Output — print tables, write JSON, command line.

Usage:

    python -m bench.repeats
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

# --- Role: settings ------------------------------------------------------------------

# Runs differ only by --thread-id; no arm is the control.
REPEATS: tuple[int, ...] = (1, 2, 3)
# Seeds whose no_llm/random scores RESULTS.md already has.
SEEDS: tuple[int, ...] = CHEAP_SEEDS
# Seed 42 of S reuses repeat 1, chosen before results.
S_REPEAT = REPEATS[0]

# Pre-registered: span median over half-width median above this.
R1_THRESHOLD = 0.5
# Published random-arm seed span from RESULTS.md, not recomputed.
RANDOM_SEED_SPAN = 0.1472


def run_dir(dataset: Dataset, repeat: int, seed: int) -> Path:
    """Folder where ``run_repeats.sh`` put one run; thread-id is its name."""
    return ARTIFACTS_DIR / f"rep{repeat}-{dataset.name}-seed{seed}"


# --- Role: thread checks ------------------------------------------------------------


def recorded_threads(winner: Winner) -> dict[str, Any] | None:
    """Thread state of the winning attempt, from its own ``result.json``.

    Not ``run_config.json``: a resume overwrites that file."""
    result = read_json_object(winner.directory / "result.json")
    if result is None:
        return None
    threads = result.get("threads")
    return threads if isinstance(threads, dict) else None


def pinned(threads: dict[str, Any] | None) -> bool | None:
    """True if all thread variables are pinned to 1; ``None`` if unrecorded."""
    if threads is None:
        return None
    if not all(key in threads for key in THREAD_ENV):
        return None
    return all(str(threads.get(key)) == "1" for key in THREAD_ENV)


def check_one_environment(threads: dict[str, dict[str, Any] | None], label: str) -> list[str]:
    """Refuse if two runs used different thread states.

    Return the names with no recorded state; those are not refused."""
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


# --- Role: R axis -------------------------------------------------------------------


def adjudicate_repeats(
    dataset: Dataset,
    resamples: int,
    seed: int = JUDGED_SEED,
    legs: dict[str, Scored] | None = None,
) -> DatasetVerdict:
    """Judge the three repeats of one dataset; a refusal becomes a verdict."""
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
    """_adjudicate_repeats | R axis: check, score, and pair the repeats."""
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
        # All three or none; two would understate the spread.
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
            # Per run, since runs may differ here.
            "threads": threads[name],
            "threads_pinned": pinned(threads[name]),
        }

    # Every pair, lower repeat first, so signs read alike.
    names = sorted(scored)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out.pairs.append(paired_delta(scored[a], scored[b], metric, resamples, seed, "repeat"))

    # Last, so a refusal leaves no legs behind.
    if legs is not None:
        legs.update(scored)


def span(result: DatasetVerdict) -> float | None:
    """Max minus min of the reproduced test scores, or ``None``."""
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


# --- Role: half widths --------------------------------------------------------------


def recorded_half_widths(seed: int = JUDGED_SEED, path: Path | None = None) -> dict[str, float]:
    """Per dataset, the median half width of published primary pairs.

    Read from ``seed42.json``, not recomputed."""
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


# --- Role: S axis -------------------------------------------------------------------


@dataclass
class SeedRow:
    """Per-seed ``llm`` test scores of one dataset, and their span."""

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
        """Max minus min score, or ``None`` if refused or too few."""
        if self.verdict != "ok" or len(self.scores) < 2:
            return None
        return max(self.scores.values()) - min(self.scores.values())


def seed_span(dataset: Dataset) -> SeedRow:
    """Collect one dataset's seed scores; a refusal becomes a row."""
    out = SeedRow(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _seed_span(dataset, out)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
    return out


def _seed_span(dataset: Dataset, out: SeedRow) -> None:
    """_seed_span | S axis: score each seed and check the rows moved."""
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
        # Same test rows under two seeds is a run bug.
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


# --- Role: verdicts -----------------------------------------------------------------


def _partial(expected: set[str], measured: list[str]) -> list[str]:
    """_partial | Verdicts: expected names that were not measured."""
    return sorted(expected - set(measured))


def r1(
    results: list[DatasetVerdict], half_widths: dict[str, float], threshold: float = R1_THRESHOLD
) -> dict[str, Any]:
    """R1 verdict: median span against median half width, binary sets only.

    Both medians use the same datasets; see ``docs/REPEATS.md``."""
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
    """R2 verdict: does any repeat pair have a CI that excludes zero?

    Pairs with identical predictions are counted apart."""
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
    """S1 observation: llm seed spans next to the random arm's span.

    ``observation_only`` is in the output so it never becomes a verdict."""
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


# --- Role: output -------------------------------------------------------------------


def report_spans(results: list[DatasetVerdict], half_widths: dict[str, float]) -> None:
    """Print the R span table."""
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
    """Print the S seed score table."""
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
    """The JSON written for this verdict."""
    return {
        "generated_by": "bench/repeats.py",
        "preregistration": "docs/REPEATS.md",
        "seed": JUDGED_SEED,
        "repeats": list(REPEATS),
        "seeds": list(SEEDS),
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # Same recorder as the runs, including cpu_count.
        "threads": thread_state(),
        # Read from RESULTS.md, not recomputed.
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
                # Derived from arms, so the two cannot disagree.
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
    """Parse the command line."""
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
    """Judge R and S, print them, write the JSON and bundle."""
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
