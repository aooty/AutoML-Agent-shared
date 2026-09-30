"""BAR-NOISE verdict: how much one setting moves at ``--margin 0.65``.

Roles:

* Cells and constants — cells, repeats, baseline, and thresholds.
* Half widths — read HARD-BAR's interval widths for N1.
* Scoring — score one cell and its pairs.
* Read back — spans, main deltas, and run shapes.
* Bar check — did each run face the predicted bar.
* Statements — the four pre-registered answers, N1 to N4.
* Output — print tables and write the verdict file.

Usage:

    python -m bench.bar_noise
    python -m bench.bar_noise adult
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

# --- Role: cells and constants ----------------------------------------------------------

# One cell per card; fixed before any repeat ran.
CELLS: tuple[tuple[str, int], ...] = (
    ("adult", 43),
    ("bank-marketing", 44),
    ("spambase", 44),
    ("speeddating", 44),
)

# Runs that differ only in ``--thread-id``.
REPEATS: tuple[int, ...] = (1, 2, 3)

# Recorded rules arm; read from HARD-BAR, never re-run.
BASELINE = "bar65nollm"

# Free check that the rules arm is deterministic.
RULES_CELL: tuple[str, int] = ("adult", 43)
RULES_REPEATS: tuple[int, ...] = (1, 2)

# Cells HARD-BAR.md judged as losses; N2 asks about these.
LOST_CELLS: tuple[tuple[str, int], ...] = (("adult", 43), ("spambase", 44))

# Same factor as R1 in REPEATS.md.
N1_THRESHOLD = 0.5

# Published REPEATS.md floor; shown by N4, never divided.
REPEATS_NOISE_FLOOR = 0.0122
REPEATS_NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트 · --margin 0.25 · 학습 1회)"

# Where N1 reads the published interval widths.
HARD_BAR_VERDICT = OUT_DIR / "hard-bar.json"

# Bootstrap seed, not a run seed; one for the file.
RESAMPLE_SEED = 42


def cell_key(dataset: str, seed: int) -> str:
    """Name of one (dataset, seed) cell."""
    return f"{dataset}-seed{seed}"


def repeat_arm(repeat: int) -> str:
    """Arm prefix of one paid repeat."""
    return f"bnoise{repeat}"


def rules_arm(repeat: int) -> str:
    """Arm prefix of one free rules repeat."""
    return f"bnoisenollm{repeat}"


def run_dir(prefix: str, dataset: str, seed: int) -> Path:
    """Archive folder of one run; thread-id is the name."""
    return ARTIFACTS_DIR / f"{prefix}-{dataset}-seed{seed}"


# --- Role: half widths ------------------------------------------------------------------


def recorded_half_widths(path: Path | None = None) -> dict[str, float]:
    """Per cell, the half width of HARD-BAR.md's main delta CI.

    Read from the verdict file, not copied here.
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


# --- Role: scoring ----------------------------------------------------------------------


def _collect(dataset: str, seed: int, metric: str, out: DatasetVerdict) -> dict[str, Winner]:
    """_collect | Role: gather repeats, baseline, and free rules legs."""
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
        # Record, do not refuse; N3 still needs the rules legs.
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
            # Missing free leg: N3 reports it, paid cells stay.
            if winner is None:
                out.missing_arms.append(arm)
            else:
                winners[arm] = winner
    return winners


def _pairs_to_take(scored: dict[str, Scored]) -> list[tuple[str, str, str]]:
    """_pairs_to_take | Role: list the comparisons this cell can make."""
    pairs: list[tuple[str, str, str]] = []
    repeats = [repeat_arm(r) for r in REPEATS if repeat_arm(r) in scored]
    # N2: each repeat against the recorded baseline.
    for arm in repeats:
        pairs.append((arm, BASELINE, "primary"))
    # N1: repeat against repeat, with intervals too.
    for i, a in enumerate(repeats):
        for b in repeats[i + 1 :]:
            pairs.append((a, b, "repeat"))
    # N3: rules arm against itself, baseline first.
    rules = [BASELINE, *(rules_arm(r) for r in RULES_REPEATS if rules_arm(r) in scored)]
    if len(rules) > 1:
        for i, a in enumerate(rules):
            for b in rules[i + 1 :]:
                pairs.append((a, b, "rules_repeat"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    """_adjudicate | Role: score one cell; raise Refusal on any mismatch."""
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
    # Refuse if runs trained with different thread counts.
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
            # Per run, since two runs may differ here.
            "threads": threads[name],
            "threads_pinned": pinned(threads[name]),
            **shapes[name],
        }

    for a_name, b_name, kind in _pairs_to_take(scored):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # Last, so a refusal leaves no legs behind.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Score one cell; a refusal empties the pairs."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=cell_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --- Role: read back --------------------------------------------------------------------


def repeat_scores(result: DatasetVerdict) -> list[float]:
    """Reproduced test scores of the repeats, in order."""
    scores: list[float] = []
    for repeat in REPEATS:
        arm = result.arms.get(repeat_arm(repeat)) or {}
        value = arm.get("test_reproduced")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append(float(value))
    return scores


def span(result: DatasetVerdict) -> float | None:
    """Max minus min of repeat scores; None if any is missing."""
    if result.verdict != "ok":
        return None
    scores = repeat_scores(result)
    if len(scores) < len(REPEATS):
        return None
    return max(scores) - min(scores)


def primary_deltas(result: DatasetVerdict) -> list[Delta]:
    """Main delta of each repeat against the baseline, in order."""
    order = {repeat_arm(r): i for i, r in enumerate(REPEATS)}
    found = [p for p in result.pairs if p.kind == "primary" and p.b == BASELINE and p.a in order]
    return sorted(found, key=lambda p: order[p.a])


def shape_agreement(result: DatasetVerdict) -> dict[str, Any]:
    """Did the repeats use the loop the same way? Recorded, not judged."""
    fields = ("fits", "stop_reason", "critic_runs")
    per_arm = {
        repeat_arm(r): {f: (result.arms.get(repeat_arm(r)) or {}).get(f) for f in fields}
        for r in REPEATS
        if repeat_arm(r) in result.arms
    }
    distinct = {json.dumps(row, sort_keys=True) for row in per_arm.values()}
    # None, not True, when some repeats are missing.
    agree = len(distinct) <= 1 if len(per_arm) == len(REPEATS) else None
    return {"per_arm": per_arm, "agree": agree}


# --- Role: bar check --------------------------------------------------------------------


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Compare each run's ``goal.threshold`` to ``hard_bar.PREDICTED_BARS``."""
    rows: list[dict[str, Any]] = []
    for dataset, seed in CELLS:
        result = results.get((dataset, seed))
        if result is None:
            continue
        predicted = PREDICTED_BARS[(dataset, seed)]
        for arm, values in result.arms.items():
            actual = values.get("goal_threshold")
            # Unrecorded bar is None, never a match.
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


# --- Role: statements -------------------------------------------------------------------


def n1(
    results: dict[tuple[str, int], DatasetVerdict],
    half_widths: dict[str, float],
    threshold: float = N1_THRESHOLD,
) -> dict[str, Any]:
    """N1: median repeat span against median HARD-BAR half width.

    Both medians use the same cells. See ``docs/BAR-NOISE.md``.
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
    """N2: do HARD-BAR's two losses repeat, three out of three?

    Identical predictions count as not judged. See ``docs/BAR-NOISE.md``.
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
        # Keep a reason for each way to be not judged.
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
    """N3: do the three rules-arm scores match within SCORE_TOLERANCE?

    Tolerance, not zero: BLAS sums drift with thread count.
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
    """N4: median span beside REPEATS.md's floor. Observation only, no ratio."""
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


# --- Role: output -----------------------------------------------------------------------


def report_spans(
    results: dict[tuple[str, int], DatasetVerdict], half_widths: dict[str, float]
) -> None:
    """Print the span table beside HARD-BAR's half widths."""
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
    """Build the verdict file body."""
    ordered = [cell for cell in CELLS if cell in results]
    return {
        "generated_by": "bench/bar_noise.py",
        "preregistration": "docs/BAR-NOISE.md",
        "measures_the_ruler_of": "docs/HARD-BAR.md",
        "margin": BAR_MARGIN,
        "repeats": list(REPEATS),
        "baseline": BASELINE,
        "run_seeds": sorted({seed for _, seed in ordered}),
        # One bootstrap seed for the whole file.
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # Same recorder the runs use, so environments compare.
        "threads": thread_state(),
        # Only widths are read; HARD-BAR is not recomputed.
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
                # Derived from arms, so the two cannot disagree.
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
        # No file: an empty one would break recheck.
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
    # Free half only: not all four statements are answered.
    incomplete = {r.dataset: r.missing_arms for r in results.values() if r.missing_arms}
    if incomplete:
        listing = "; ".join(f"{key}: {', '.join(arms)}" for key, arms in incomplete.items())
        print(f"아직 돌지 않은 팔이 있습니다 — {listing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
