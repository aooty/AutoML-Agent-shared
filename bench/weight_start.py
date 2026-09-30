"""Does the fixed weight ladder pay off in the loop, and does the headline move?

Roles:

* Arms and constants — arm names, pinned predictions, published deltas.
* Gate checks — first weight rung, verdict source, and bar.
* Scoring — score one (dataset, seed) cell.
* Verdict A — does the new weight prescription pay off.
* Verdict B — re-judge the headline with HARD-BAR's rule.
* Output — build the JSON payload and print the report.

Usage:

    python -m bench.weight_start
    python -m bench.weight_start adult --seeds 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench import hard_bar
from bench.datasets import BY_NAME, JUDGED_SEED
from bench.hard_bar import BAR_TOLERANCE, DATASETS, MARGIN, PREDICTED_BARS, SEEDS
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
    random_winner,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.random_search import run_dir as random_run_dir
from bench.replan import first_iteration_winner, run_shape

# --- Role: arms and constants -----------------------------------------------------------

# New prefix, so recorded archives are never written to.
FIXED = "bar65nollmratio"
RULES = "bar65nollm"
LOOP = "bar65llm"

# The one commit that separates FIXED from RULES.
LEVER_COMMIT = "d37217e"

# Predicted first weight rung: (iteration, weight). None means no weight.
# Written by hand on purpose; a test checks it still holds.
EXPECTED_FIRST_RUNG: dict[str, tuple[int, float] | None] = {
    "adult": (1, 3.179),
    "bank-marketing": (1, 7.547),
    "speeddating": (2, 5.072),
    "spambase": None,
}

# What the old code did, from the recorded archive.
RECORDED_FIRST_RUNG: dict[str, tuple[int, float] | None] = {
    "adult": (1, 1.5),
    "bank-marketing": (1, 1.5),
    "speeddating": (2, 1.5),
    "spambase": None,
}

# Expected verdict source and count in each recorded arm.
EXPECTED_SOURCES: dict[str, tuple[str, int]] = {
    LOOP: ("llm", 37),
    RULES: ("heuristic", 43),
}

# Main deltas from HARD-BAR.md, only to cross-check ours.
PRIMARY_DELTAS: dict[tuple[str, int], float] = {
    ("speeddating", 42): 0.0819,
    ("speeddating", 43): 0.0777,
    ("speeddating", 44): 0.0761,
}
PUBLISHED_TOLERANCE = 5e-5

# Verdict B uses hard_bar's rule, so use its floor.
NOISE_FLOOR = hard_bar.NOISE_FLOOR
NOISE_FLOOR_SOURCE = hard_bar.NOISE_FLOOR_SOURCE

# Bootstrap seed, not a run seed; one for the file.
RESAMPLE_SEED = JUDGED_SEED

# Where verdict A checks recovery and identity.
POSITIVE_ON = "speeddating"
IDENTITY_ON = "spambase"


def _key(dataset: str, seed: int) -> str:
    """_key | Role: name of one (dataset, seed) cell."""
    return f"{dataset}-seed{seed}"


def run_directory(arm: str, dataset: str, seed: int) -> Path:
    """Archive folder of one run."""
    return ARTIFACTS_DIR / f"{arm}-{dataset}-seed{seed}"


# --- Role: gate checks ------------------------------------------------------------------


def prescriptions(directory: Path) -> list[dict[str, Any]]:
    """All Critic verdicts of one run, read from history.json."""
    history = read_json_object(directory / "history.json") or {}
    attempts = history.get("history") or []
    out: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        critic = attempt.get("critic")
        if not isinstance(critic, dict):
            continue
        changes = critic.get("concrete_changes")
        weight = changes.get("class_weight") if isinstance(changes, dict) else None
        out.append(
            {
                "iteration": attempt.get("iteration"),
                "source": critic.get("source"),
                "failure_type": critic.get("failure_type"),
                "class_weight": weight,
            }
        )
    return out


def positive_weight(weight: Any) -> float | None:
    """Positive-class weight from a ``class_weight`` map, else None.

    Keys are string class codes, so read ``"1"``.
    """
    if not isinstance(weight, dict):
        return None
    value = weight.get("1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def first_rung(directory: Path) -> tuple[int, float] | None:
    """First ``class_weight`` the Critic gave: (iteration, weight)."""
    for row in prescriptions(directory):
        value = positive_weight(row["class_weight"])
        if value is not None and isinstance(row["iteration"], int):
            return (row["iteration"], value)
    return None


def first_rung_check(arm: str = FIXED) -> list[dict[str, Any]]:
    """Gate 1: did the first rung land where EXPECTED_FIRST_RUNG says?"""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = run_directory(arm, dataset, seed)
            expected = EXPECTED_FIRST_RUNG[dataset]
            present = (directory / "history.json").exists()
            actual = first_rung(directory) if present else None
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "arm": arm,
                    "expected_iteration": None if expected is None else expected[0],
                    "expected_weight": None if expected is None else expected[1],
                    "actual_iteration": None if actual is None else actual[0],
                    "actual_weight": None if actual is None else actual[1],
                    # A missing run is not a match for None.
                    "run_present": present,
                    "matches": present and actual == expected,
                }
            )
    return rows


def source_check() -> list[dict[str, Any]]:
    """Gate 2: where each arm's Critic verdicts came from."""
    rows: list[dict[str, Any]] = []
    for arm, (expected_source, expected_count) in (
        *EXPECTED_SOURCES.items(),
        (FIXED, ("heuristic", 0)),
    ):
        counts: dict[str, int] = {}
        present = 0
        for dataset in DATASETS:
            for seed in SEEDS:
                directory = run_directory(arm, dataset, seed)
                if not (directory / "history.json").exists():
                    continue
                present += 1
                for row in prescriptions(directory):
                    key = str(row["source"])
                    counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        rows.append(
            {
                "arm": arm,
                "runs_present": present,
                "sources": counts,
                "expected_source": expected_source,
                # 0 means not pinned; new arm count is unknown.
                "expected_count": expected_count or None,
                "matches": (
                    present == len(DATASETS) * len(SEEDS)
                    and set(counts) <= {expected_source}
                    and total > 0
                    and (expected_count == 0 or total == expected_count)
                ),
            }
        )
    return rows


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Gate 3: did the new runs face the HARD-BAR.md bars?"""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            predicted = PREDICTED_BARS[(dataset, seed)]
            actual = result.arms.get(FIXED, {}).get("goal_threshold")
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
                    "arm": FIXED,
                    "predicted_bar": predicted,
                    "actual_bar": actual,
                    "off_by": off,
                    "matches": off is not None and off <= BAR_TOLERANCE,
                }
            )
    return rows


# --- Role: scoring ----------------------------------------------------------------------


def _legs(
    dataset: str, seed: int, metric: str, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, dict[str, Any]]]:
    """_legs | Role: find each arm's winner and run shape."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm in (FIXED, RULES, LOOP):
        directory = run_directory(arm, dataset, seed)
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    if FIXED in winners:
        # Also score the new arm at iteration 1.
        winners[f"{FIXED}@it1"] = first_iteration_winner(
            FIXED, run_directory(FIXED, dataset, seed), metric
        )
    return winners, shapes


def _pairs_to_take(scored: dict[str, Scored], fits: int) -> list[tuple[str, str, str]]:
    """_pairs_to_take | Role: list the comparisons this cell can make."""
    pairs: list[tuple[str, str, str]] = []
    if FIXED in scored and RULES in scored:
        pairs.append((FIXED, RULES, "prescription"))
    if LOOP in scored and FIXED in scored:
        pairs.append((LOOP, FIXED, "headline"))
    if LOOP in scored and RULES in scored:
        # Needed for the half-bar and the published check.
        pairs.append((LOOP, RULES, "main"))
    if FIXED in scored and f"{FIXED}@it1" in scored:
        pairs.append((FIXED, f"{FIXED}@it1", "ratio_replan_gain"))
    matched = f"random@k={fits}"
    if FIXED in scored and matched in scored:
        pairs.append((FIXED, matched, "ratio_matched_level"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    """_adjudicate | Role: score one cell; raise Refusal on any mismatch."""
    entry = BY_NAME[dataset]
    metric = entry.metric
    winners, shapes = _legs(dataset, seed, metric, out)
    if FIXED not in winners:
        # The run has not happened yet.
        raise Refusal(
            f"{dataset} 시드 {seed}: {FIXED} 실행이 없습니다 — "
            "bench/scripts/run_weight_start.sh를 먼저 돌려야 판정할 수 있습니다"
        )

    random_dir = random_run_dir(entry, seed)
    if (random_dir / "summary.json").exists():
        fits = winners[FIXED].fits
        if fits < 1:
            raise Refusal(f"{dataset} 시드 {seed}: {FIXED}의 학습 횟수가 {fits}회로 기록됐습니다")
        winners[f"random@k={fits}"] = random_winner(random_dir, fits, metric)
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # Iteration 1 has no recorded test score (nan).
        if not math.isnan(winner.recorded_test):
            gap = abs(result.test_score - winner.recorded_test)
            if gap > SCORE_TOLERANCE:
                raise Refusal(
                    f"{name}: 다시 유도한 test 점수 {result.test_score:.6f}가 기록된 "
                    f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
                )
        scored[name] = result

    fingerprints = {name: s.fingerprint for name, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{name}={fp[:12]}" for name, fp in sorted(fingerprints.items()))
        raise Refusal(
            f"{dataset} 시드 {seed}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다"
        )
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset} 시드 {seed}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )

    any_scored = next(iter(scored.values()))
    out.test_rows = any_scored.n_rows
    out.test_fingerprint = any_scored.fingerprint
    for name, s in scored.items():
        config_weight = dict(s.winner.config.get("hyperparams") or {}).get("class_weight")
        out.arms[name] = {
            "label": s.winner.label,
            "fits": s.winner.fits,
            "val_score": s.winner.val_score,
            "test_recorded": s.winner.recorded_test,
            "test_reproduced": s.test_score,
            "selection_gap": s.winner.val_score - s.test_score,
            "class_weight": config_weight,
            "model": s.winner.config.get("model"),
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    for a_name, b_name, kind in _pairs_to_take(scored, winners[FIXED].fits):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # Last, so a refusal leaves no legs behind.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Score one (dataset, seed); a refusal empties the pairs."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


def _pair(result: DatasetVerdict, kind: str) -> Delta | None:
    """_pair | Role: find the pair of one kind."""
    return next((p for p in result.pairs if p.kind == kind), None)


def published_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Compare our main deltas to the ones HARD-BAR.md printed."""
    rows: list[dict[str, Any]] = []
    for (dataset, seed), published in sorted(PRIMARY_DELTAS.items()):
        result = results.get((dataset, seed))
        pair = None if result is None else _pair(result, "main")
        recomputed = None if pair is None else pair.delta
        off = None if recomputed is None else abs(round(recomputed, 4) - published)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "published": published,
                "recomputed": recomputed,
                "off_by": off,
                "matches": off is not None and off <= PUBLISHED_TOLERANCE,
            }
        )
    return rows


def identity_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Per cell: is the fixed arm's run the same as the recorded one?"""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            pair = None if result is None else _pair(result, "prescription")
            fixed = {} if result is None else result.arms.get(FIXED, {})
            rules = {} if result is None else result.arms.get(RULES, {})
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "identical_predictions": None if pair is None else pair.identical_predictions,
                    "delta": None if pair is None else pair.delta,
                    "same_fits": fixed.get("fits") == rules.get("fits"),
                    "same_stop_reason": fixed.get("stop_reason") == rules.get("stop_reason"),
                    "same_winner_label": fixed.get("label") == rules.get("label"),
                    "predicted_identical": EXPECTED_FIRST_RUNG[dataset] is None,
                }
            )
    return rows


# --- Role: verdict A --------------------------------------------------------------------


def _cell(result: DatasetVerdict | None) -> dict[str, Any]:
    """_cell | Role: reduce one cell to what verdict A reads."""
    if result is None or result.verdict != "ok":
        return {"evaluated": False, "reason": "없음 또는 거부"}
    prescription = _pair(result, "prescription")
    main = _pair(result, "main")
    if prescription is None or main is None:
        return {"evaluated": False, "reason": "쌍이 모자랍니다"}
    half = main.delta / 2.0
    identical = prescription.identical_predictions
    return {
        "evaluated": True,
        "main": main.delta,
        "half_of_main": half,
        "prescription": prescription.delta,
        "prescription_ci": [prescription.ci_low, prescription.ci_high],
        "identical_predictions": identical,
        # Condition 1: CI above zero and at least half.
        "clears_zero": (not identical) and prescription.ci_low > 0,
        "reaches_half": prescription.delta >= half,
        # Condition 2: identical, not just close.
        "identical_to_recorded": identical,
        # Condition 3: no cell may lose.
        "below_zero": (not identical) and prescription.ci_high < 0,
    }


def prescription_verdict(
    results: dict[tuple[str, int], DatasetVerdict], gates: dict[str, bool]
) -> dict[str, Any]:
    """Verdict A, as pre-registered in ``docs/WEIGHT-START.md``.

    Order: gates first, then missing cells, then conditions.
    """
    cells = {
        (dataset, seed): _cell(results.get((dataset, seed)))
        for dataset in DATASETS
        for seed in SEEDS
    }
    absent = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if not cell["evaluated"]
    ]
    failed_gates = [name for name, ok in gates.items() if not ok]

    def _all(dataset: str, field: str) -> bool:
        return all(
            cells[(dataset, seed)]["evaluated"] and cells[(dataset, seed)][field] for seed in SEEDS
        )

    recovered = _all(POSITIVE_ON, "clears_zero") and _all(POSITIVE_ON, "reaches_half")
    identical = _all(IDENTITY_ON, "identical_to_recorded")
    losses = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if cell["evaluated"] and cell["below_zero"]
    ]
    no_loss = not losses

    if failed_gates:
        verdict = "판정 불가"
        status = (
            f"판정 불가 — 전제 검사가 어긋났습니다 ({', '.join(failed_gates)}). "
            "코드 변경이 예상과 다른 자리에 착지했거나 실행 조건이 다릅니다 — "
            "기준을 약하게 고치지 않습니다"
        )
    elif absent:
        verdict = "부분 판정"
        status = (
            f"부분 판정 — (데이터셋, 시드) {len(DATASETS) * len(SEEDS)}칸 중 판정하지 못한 칸이 "
            f"있습니다 (없음/거부: {', '.join(absent)})"
        )
    elif recovered and identical and no_loss:
        verdict = "처방이 산다"
        status = (
            f"처방이 산다 — {POSITIVE_ON} 세 시드 전부에서 고친 사다리가 주 Δ의 절반 이상을 "
            f"회수했고, {IDENTITY_ON}는 예측까지 동일하고, 짐이 된 칸이 없습니다"
        )
    elif not recovered:
        verdict = "처방이 죽는다"
        status = (
            f"처방이 죽는다 — {POSITIVE_ON}에서 첫 단을 카드 비율로 옮긴 코드가 루프에서 주 Δ의 "
            "절반을 회수하지 못했습니다. WEIGHT-LEVER가 본 회수는 루프가 도달할 수 있는 설정이 "
            "아니었습니다"
        )
    else:
        broken = [
            name for name, ok in (("동일성", identical), ("짐 없음", no_loss)) if not ok
        ]
        verdict = "부분 지지"
        status = (
            f"부분 지지 — 회수는 세 시드 전부에서 됐지만 {', '.join(broken)} 조건이 깨졌습니다"
        )
    if losses and verdict not in ("판정 불가", "부분 판정"):
        status += f" / CI가 0 아래인 칸: {', '.join(losses)}"

    return {
        "treatment": FIXED,
        "baseline": RULES,
        "lever_commit": LEVER_COMMIT,
        "scope_datasets": list(DATASETS),
        "scope_seeds": list(SEEDS),
        "positive_prediction_on": POSITIVE_ON,
        "identity_prediction_on": IDENTITY_ON,
        "resample_seed": RESAMPLE_SEED,
        "expected_first_rung": {d: list(v) if v else None for d, v in EXPECTED_FIRST_RUNG.items()},
        "recorded_first_rung": {d: list(v) if v else None for d, v in RECORDED_FIRST_RUNG.items()},
        "gates": gates,
        "failed_gates": failed_gates,
        "cells": {f"{d}-seed{s}": cell for (d, s), cell in cells.items()},
        "condition_1_recovery": recovered,
        "condition_2_identical": identical,
        "condition_3_no_loss": no_loss,
        "below_zero_cells": losses,
        "not_evaluated": absent,
        "verdict": verdict,
        "status": status,
    }


# --- Role: verdict B --------------------------------------------------------------------


def as_hard_bar_board(
    results: dict[tuple[str, int], DatasetVerdict],
) -> dict[tuple[str, int], DatasetVerdict]:
    """Rename our cells so ``hard_bar.criterion`` can judge them."""
    board: dict[tuple[str, int], DatasetVerdict] = {}
    for (dataset, seed), result in results.items():
        shim = DatasetVerdict(
            dataset=result.dataset,
            metric=result.metric,
            task=result.task,
            verdict=result.verdict,
            refusal=result.refusal,
            test_rows=result.test_rows,
            test_fingerprint=result.test_fingerprint,
        )
        loop_row = dict(result.arms.get(LOOP, {}))
        fixed_row = dict(result.arms.get(FIXED, {}))
        shim.arms[hard_bar.PRIMARY[0]] = loop_row
        shim.arms[hard_bar.PRIMARY[1]] = fixed_row
        headline = _pair(result, "headline")
        if headline is not None:
            shim.pairs.append(
                Delta(
                    a=hard_bar.PRIMARY[0],
                    b=hard_bar.PRIMARY[1],
                    kind="primary",
                    delta=headline.delta,
                    ci_low=headline.ci_low,
                    ci_high=headline.ci_high,
                    p_better=headline.p_better,
                    resamples_used=headline.resamples_used,
                    identical_predictions=headline.identical_predictions,
                )
            )
        board[(dataset, seed)] = shim
    return board


def headline_verdict(
    results: dict[tuple[str, int], DatasetVerdict], gates: dict[str, bool]
) -> dict[str, Any]:
    """Verdict B: HARD-BAR.md's rule, unchanged, against the fixed arm."""
    outcome = hard_bar.criterion(as_hard_bar_board(results))
    failed_gates = [name for name, ok in gates.items() if not ok]
    if failed_gates:
        outcome["complete"] = False
        outcome["status"] = (
            f"판정 불가 — 전제 검사가 어긋났습니다 ({', '.join(failed_gates)}). "
            f"원래 상태: {outcome['status']}"
        )
    # Fix the arm names hard_bar filled from PRIMARY.
    outcome["challenger"] = LOOP
    outcome["baseline"] = FIXED
    outcome["criterion_from"] = "bench/hard_bar.py criterion (글자 그대로 재사용)"
    outcome["compared_against"] = "docs/HARD-BAR.md의 구분되지 않음 (이김 1 / 그 밖 1 / 짐 2)"
    outcome["gates"] = gates
    outcome["failed_gates"] = failed_gates
    return outcome


# --- Role: output -----------------------------------------------------------------------


def gate_status(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, bool]:
    """Reduce the three gates to booleans."""
    return {
        "first_rung": all(row["matches"] for row in first_rung_check()),
        "verdict_sources": all(row["matches"] for row in source_check()),
        "bars": bool(results) and all(row["matches"] for row in bar_check(results)),
    }


def budget_used(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """What each arm spent per cell."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            row: dict[str, Any] = {"dataset": dataset, "seed": seed}
            for arm in (FIXED, RULES, LOOP):
                arm_row = result.arms.get(arm, {})
                row[arm] = {
                    "fits": arm_row.get("fits"),
                    "stop_reason": arm_row.get("stop_reason"),
                    "critic_runs": arm_row.get("critic_runs"),
                    "winner": arm_row.get("label"),
                    "model": arm_row.get("model"),
                    "class_weight": arm_row.get("class_weight"),
                    "val_score": arm_row.get("val_score"),
                    "test_reproduced": arm_row.get("test_reproduced"),
                }
            rows.append(row)
    return rows


def payload(results: dict[tuple[str, int], DatasetVerdict], resamples: int) -> dict[str, Any]:
    """Build the verdict file body."""
    ordered = [
        (dataset, seed) for dataset in DATASETS for seed in SEEDS if (dataset, seed) in results
    ]
    gates = gate_status(results)
    return {
        "generated_by": "bench/weight_start.py",
        "preregistration": "docs/WEIGHT-START.md",
        "acts_on": "docs/WEIGHT-LEVER.md",
        "re_judges": "docs/HARD-BAR.md",
        "lever_commit": LEVER_COMMIT,
        "margin": MARGIN,
        "reuses_recorded_arms": [RULES, LOOP],
        "test_rows_shared_with": "docs/HARD-BAR.md",
        "run_seeds": sorted({seed for _, seed in ordered}),
        # One bootstrap seed for the whole file.
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
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
                "arms": results[cell].arms,
                "pairs": [vars(p) for p in results[cell].pairs],
            }
            for cell in ordered
        ],
        "gates": gates,
        "first_rung_check": first_rung_check(),
        "source_check": source_check(),
        "bar_check": bar_check(results),
        "identity_check": identity_check(results),
        "published_check": published_check(results),
        "budget": budget_used(results),
        "noise_floor": NOISE_FLOOR,
        "noise_floor_source": NOISE_FLOOR_SOURCE,
        "prescription": prescription_verdict(results, gates),
        "headline": headline_verdict(results, gates),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "고친 사다리가 루프에서 값을 내는가, 그리고 헤드라인 판정이 바뀌는가 "
            "(docs/WEIGHT-START.md의 사전 등록 기준)"
        )
    )
    parser.add_argument("names", nargs="*", help=f"데이터셋 이름 (생략하면 {' '.join(DATASETS)})")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help=f"기본: {' '.join(str(s) for s in SEEDS)} (그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/weight-start.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.names or list(DATASETS)
    unknown = [name for name in names if name not in DATASETS] + [
        str(s) for s in args.seeds if s not in SEEDS
    ]
    if unknown:
        print(
            f"사전 등록의 범위 밖: {', '.join(unknown)} "
            f"(데이터셋 {', '.join(DATASETS)} · 시드 {', '.join(str(s) for s in SEEDS)})",
            file=sys.stderr,
        )
        return 2

    print(
        f"고친 사다리의 판정 · 레버 {LEVER_COMMIT} · margin {MARGIN} · "
        f"재추출 {args.resamples}회 (재추출 시드 {RESAMPLE_SEED}) · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: dict[tuple[str, int], DatasetVerdict] = {}
    for dataset in DATASETS:
        if dataset not in names:
            continue
        for seed in SEEDS:
            if seed not in args.seeds:
                continue
            collected: dict[str, Scored] = {}
            results[(dataset, seed)] = adjudicate(dataset, seed, args.resamples, collected)
            if collected:
                legs[results[(dataset, seed)].dataset] = collected
    report(list(results.values()))

    out_path = args.out or OUT_DIR / "weight-start.json"
    if not legs:
        # No file: an empty one would break recheck.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_weight_start.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    body = payload(results, args.resamples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results.values()})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")

    print()
    print("첫 단 — 예측 대조:")
    for row in first_rung_check():
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} "
            f"예측=(it{row['expected_iteration']}, {row['expected_weight']}) "
            f"실측=(it{row['actual_iteration']}, {row['actual_weight']}) {mark}"
        )
    print()
    print("verdict 출처:")
    for row in source_check():
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['arm']:<16} 실행={row['runs_present']} 출처={row['sources']} "
            f"기대={row['expected_source']} {mark}"
        )
    print()
    print(f"바 — 예측 대조 (허용 {BAR_TOLERANCE:g}):")
    for row in bar_check(results):
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} 예측={row['predicted_bar']:.4f} "
            f"실측={row['actual_bar']} {mark}"
        )
    print()
    print(f"{IDENTITY_ON} 동일성 (예측: 예측까지 동일):")
    for row in identity_check(results):
        if not row["predicted_identical"]:
            continue
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} 동일예측={row['identical_predictions']} "
            f"Δ={row['delta']} 같은학습={row['same_fits']} 같은종료={row['same_stop_reason']} "
            f"같은우승={row['same_winner_label']}"
        )
    print()
    print("예산 — 실제로 쓴 것:")
    for row in budget_used(results):
        for arm in (FIXED, RULES, LOOP):
            spent = row[arm]
            print(
                f"   {row['dataset']:16s} 시드 {row['seed']} {arm:<16} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"가중치={spent['class_weight']} 우승={spent['winner']}"
            )
    print()
    print(f"판정 A (처방): {body['prescription']['status']}")
    print(f"판정 B (헤드라인): {body['headline']['status']}")
    print(f"잡음 바닥 {NOISE_FLOOR} — {NOISE_FLOOR_SOURCE}")
    print(f"고친 규칙 팔 시드 산포: {body['headline']['rules_seed_spread']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results.values() if r.verdict != "ok"]
    if refused:
        print(f"거부된 칸: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
