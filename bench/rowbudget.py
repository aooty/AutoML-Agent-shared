"""Judge docs/ROWBUDGET.md: does the row budget prompt section change plans?

Roles:

* Run paths — where each arm's runs are.
* P1 plan check — did the first plans differ?
* P2 forecast check — prompt row counts against runner counts.
* One dataset — read both arms and judge them.
* Preregistered claims — sum up P1, P2 and P3.
* Output — print the table, write JSON, CLI.

Usage:
    python -m bench.rowbudget
    python -m bench.rowbudget spambase
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automl_agent.capabilities import (
    DEFAULT_VALIDATION_FRACTION,
    EARLY_STOPPING_AUTO_MIN_ROWS,
    SELF_VALIDATING_MODELS,
)
from automl_agent.config import read_json_object
from automl_agent.scoring.splits import row_counts
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import ALL, BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    Delta,
    Refusal,
    Scored,
    Winner,
    loop_winner,
    paired_delta,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.repeats import check_one_environment, pinned, recorded_threads

# --- Role: run paths --------------------------------------------------------------------

# Control first, so a positive P3 delta means the section helped.
ARMS: tuple[tuple[str, str], ...] = (("rb_off", "rboff"), ("rb_on", "rbon"))

# Preregistered keys only; model swaps show through capacity keys.
PLAN_KEYS: tuple[str, ...] = ("early_stopping", "validation_fraction", "n_estimators", "max_iter")

# The first plan, written before any result existed.
FIRST_ITERATION = 1


def run_dir(dataset: Dataset, prefix: str, seed: int = JUDGED_SEED) -> Path:
    return ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}"


def first_iteration_dir(dataset: Dataset, prefix: str, seed: int = JUDGED_SEED) -> Path:
    return run_dir(dataset, prefix, seed) / "train" / f"iter_{FIRST_ITERATION:02d}"


# --- Role: P1 plan check ----------------------------------------------------------------

# None means the plan left this key out.
ABSENT = None


@dataclass
class Plan:
    """One arm's first-iteration plan, cut down to the P1 keys."""

    model: str | None
    keys: dict[str, Any]
    directory: Path

    @classmethod
    def read(cls, directory: Path, arm: str) -> Plan:
        config = read_json_object(directory / "train_config.json")
        if config is None:
            raise Refusal(
                f"{arm}: iteration {FIRST_ITERATION}의 train_config.json을 읽을 수 없습니다 ({directory})"
            )
        hyperparams = config.get("hyperparams")
        if not isinstance(hyperparams, dict):
            raise Refusal(f"{arm}: train_config.json에 hyperparams 블록이 없습니다 ({directory})")
        return cls(
            model=str(config.get("model")) if config.get("model") is not None else None,
            keys={key: hyperparams.get(key, ABSENT) for key in PLAN_KEYS},
            directory=directory,
        )


def plan_difference(off: Plan, on: Plan) -> dict[str, Any]:
    """Preregistered keys the two arms set differently.

    Reads the plan, not ``applied_hyperparams``; the model is noted beside it."""
    differing = sorted(
        key for key in PLAN_KEYS if off.keys.get(key, ABSENT) != on.keys.get(key, ABSENT)
    )
    return {
        "differs": bool(differing),
        "differing_keys": differing,
        "model": {"rb_off": off.model, "rb_on": on.model},
        "model_differs": off.model != on.model,
        "keys": {"rb_off": off.keys, "rb_on": on.keys},
    }


# --- Role: P2 forecast check ------------------------------------------------------------


@dataclass
class Forecast:
    """What the prompt section told the planner, recomputed from the card."""

    card_rows: int
    train: int
    auto_on: bool
    held_out_at_default: int


def forecast(card: dict[str, Any]) -> Forecast:
    """Redo the section's row math from the card's ``n_rows`` with ``row_counts``."""
    n_rows: Any = card.get("n_rows")
    try:
        counts = row_counts(int(n_rows))
    except (TypeError, ValueError) as exc:
        raise Refusal(f"카드의 n_rows를 읽을 수 없습니다 ({n_rows!r}): {exc}") from exc
    train = counts["train"]
    return Forecast(
        card_rows=int(n_rows),
        train=train,
        auto_on=train > EARLY_STOPPING_AUTO_MIN_ROWS,
        held_out_at_default=math.ceil(DEFAULT_VALIDATION_FRACTION * train),
    )


def check_forecast(predicted: Forecast, result: dict[str, Any], arm: str, dataset: str) -> dict[str, Any]:
    """P2 for one run: forecast rows against ``internal_validation``.

    Checks held-out and total train rows; a run holding nothing out is noted."""
    internal = result.get("internal_validation")
    if not isinstance(internal, dict):
        applied = result.get("applied_hyperparams")
        model_note = ""
        if isinstance(applied, dict) and applied.get("early_stopping") is False:
            model_note = " (계획이 early_stopping을 false로 껐습니다)"
        return {
            "arm": arm,
            "checked": False,
            "reason": (
                "이 실행은 행을 떼어 두지 않았습니다 — internal_validation이 없으므로 "
                f"예고와 대조할 것이 없습니다{model_note}"
            ),
            "forecast_train": predicted.train,
        }
    held_out = internal.get("held_out_rows")
    fit_rows = internal.get("fit_rows")
    fraction = internal.get("validation_fraction", DEFAULT_VALIDATION_FRACTION)
    if not isinstance(held_out, int) or not isinstance(fit_rows, int):
        raise Refusal(
            f"{dataset}/{arm}: internal_validation의 행 수가 정수가 아닙니다 ({internal!r})"
        )
    expected_held_out = math.ceil(float(fraction) * predicted.train)
    measured_train = held_out + fit_rows
    problems: list[str] = []
    if measured_train != predicted.train:
        problems.append(
            f"학습 행 수: 예고 {predicted.train:,} vs 실측 {measured_train:,} "
            f"(held_out {held_out:,} + fit {fit_rows:,})"
        )
    if held_out != expected_held_out:
        problems.append(
            f"떼어 둔 행 수: validation_fraction {fraction:g}에서 예고 {expected_held_out:,} vs "
            f"실측 {held_out:,}"
        )
    return {
        "arm": arm,
        "checked": True,
        "forecast_train": predicted.train,
        "forecast_held_out_at_default": predicted.held_out_at_default,
        "measured_train": measured_train,
        "measured_held_out": held_out,
        "measured_fit": fit_rows,
        "validation_fraction": fraction,
        "expected_held_out": expected_held_out,
        "matches": not problems,
        "problems": problems,
    }


# --- Role: one dataset ------------------------------------------------------------------


@dataclass
class Row:
    dataset: str
    metric: str
    task: str
    verdict: str = "ok"
    refusal: str | None = None
    test_rows: int = 0
    test_fingerprint: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)
    p2: list[dict[str, Any]] = field(default_factory=list)
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    delta: Delta | None = None
    missing_arms: list[str] = field(default_factory=list)
    caveat: str | None = None


def adjudicate(
    dataset: Dataset,
    resamples: int,
    seed: int = JUDGED_SEED,
    legs: dict[str, Scored] | None = None,
) -> Row:
    out = Row(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, resamples, seed, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.delta = None
    return out


def _adjudicate(
    dataset: Dataset, resamples: int, seed: int, out: Row, legs: dict[str, Scored] | None = None
) -> None:
    """_adjudicate | One dataset: fill ``out`` or raise Refusal."""
    metric = dataset.metric
    winners: dict[str, Winner] = {}
    for arm, prefix in ARMS:
        winner = loop_winner(arm, run_dir(dataset, prefix, seed), metric)
        if winner is None:
            out.missing_arms.append(arm)
        else:
            winners[arm] = winner
    if out.missing_arms:
        raise Refusal(
            f"{dataset.name}: {', '.join(out.missing_arms)} 실행이 없습니다 — "
            "이 판정은 두 팔이 다 있어야 성립합니다"
        )

    # Read iteration 1, not the winner, in case of extra iterations.
    plans = {
        arm: Plan.read(first_iteration_dir(dataset, prefix, seed), arm) for arm, prefix in ARMS
    }
    out.plan = plan_difference(plans["rb_off"], plans["rb_on"])

    # Both arms got the same card; P2 checks against it.
    card = read_json_object(dataset.card_path(seed))
    if card is None:
        raise Refusal(f"{dataset.name}: 카드를 읽을 수 없습니다 ({dataset.card_path(seed)})")
    predicted = forecast(card)
    out.forecast = {
        "card_rows": predicted.card_rows,
        "train": predicted.train,
        "early_stopping_auto_on": predicted.auto_on,
        "held_out_at_default_fraction": predicted.held_out_at_default,
        "self_validating_models": list(SELF_VALIDATING_MODELS),
    }
    for arm, prefix in ARMS:
        result = read_json_object(first_iteration_dir(dataset, prefix, seed) / "result.json")
        if result is None:
            raise Refusal(f"{dataset.name}/{arm}: iteration {FIRST_ITERATION}의 result.json이 없습니다")
        out.p2.append(check_forecast(predicted, result, arm, dataset.name))

    # P3 is observation only, built from paired.py's tools.
    scored: dict[str, Scored] = {}
    threads: dict[str, dict[str, Any] | None] = {}
    for arm, winner in winners.items():
        rescored = reproduce(winner, metric)
        gap = abs(rescored.test_score - winner.recorded_test)
        if gap > SCORE_TOLERANCE:
            raise Refusal(
                f"{arm}: 다시 유도한 test 점수 {rescored.test_score:.6f}가 기록된 "
                f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
            )
        scored[arm] = rescored
        threads[arm] = recorded_threads(winner)

    fingerprints = {arm: s.fingerprint for arm, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{arm}={fp[:12]}" for arm, fp in sorted(fingerprints.items()))
        raise Refusal(f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {
        arm: json.dumps(split_identity(w.config), sort_keys=True) for arm, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset.name}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )
    unrecorded = check_one_environment(threads, dataset.name)
    if unrecorded:
        out.caveat = (
            f"{', '.join(unrecorded)}의 스레드 상태가 기록에 없습니다 — "
            "고정이 지켜졌는지 산출물로는 확인할 수 없습니다"
        )

    any_scored = next(iter(scored.values()))
    out.test_rows = any_scored.n_rows
    out.test_fingerprint = any_scored.fingerprint
    for arm, s in scored.items():
        out.arms[arm] = {
            "label": s.winner.label,
            "fits": s.winner.fits,
            "val_score": s.winner.val_score,
            "test_recorded": s.winner.recorded_test,
            "test_reproduced": s.test_score,
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            "threads": threads[arm],
            "threads_pinned": pinned(threads[arm]),
        }
    out.delta = paired_delta(scored["rb_on"], scored["rb_off"], metric, resamples, seed, "observation")

    # Last, so a refused dataset leaves no legs behind.
    if legs is not None:
        legs.update(scored)


# --- Role: preregistered claims ---------------------------------------------------------


def p1(rows: list[Row], threshold: int = 2) -> dict[str, Any]:
    """P1: count binary datasets whose first plans differ.

    Two or more means the section changes plans; one is reported as is."""
    expected = set(JUDGED_NAMES)
    binary = [r for r in rows if r.dataset in expected]
    changed = sorted(r.dataset for r in binary if r.verdict == "ok" and r.plan.get("differs"))
    same = sorted(r.dataset for r in binary if r.verdict == "ok" and r.plan and not r.plan.get("differs"))
    measured = sorted({*changed, *same})
    missing = sorted(expected - set(measured))
    tally = f"달라짐 {len(changed)} / 같음 {len(same)}"
    if not measured:
        status = f"P1 판정 안 함 — 계획을 비교한 이진 데이터셋이 없습니다 (없음: {', '.join(missing)})"
    elif missing:
        status = (
            f"P1 부분 판정 — 기준은 이진 {len(expected)}개에 대한 것인데 {len(measured)}개만 "
            f"비교했습니다 (없음: {', '.join(missing)}). 지금까지 {tally}"
        )
    elif len(changed) >= threshold:
        status = (
            f"P1 충족 — 이 절은 계획을 바꿉니다 ({len(changed)}개: {', '.join(changed)}). {tally}"
        )
    elif not changed:
        status = (
            "P1 미충족 — 프롬프트에 넣었지만 계획은 같았습니다. 이 절은 프롬프트 4줄의 값을 "
            f"못 했습니다 ({tally})"
        )
    else:
        status = (
            f"P1 경계 — 달라진 데이터셋이 {len(changed)}개({', '.join(changed)})로, 사전 등록이 "
            f"이름을 붙여 둔 두 경우(2개 이상 / 0개) 사이입니다. {tally}"
        )
    return {
        "threshold": threshold,
        "expected_binary": sorted(expected),
        "measured": measured,
        "changed": changed,
        "unchanged": same,
        "not_evaluated": missing,
        "per_dataset": {r.dataset: r.plan for r in rows if r.plan},
        "complete": not missing,
        "status": status,
    }


def p2(rows: list[Row]) -> dict[str, Any]:
    """P2: any mismatch in either arm is a bug and halts."""
    checks = [check for row in rows for check in row.p2]
    mismatched = [check for check in checks if check.get("checked") and not check.get("matches")]
    checked = [check for check in checks if check.get("checked")]
    unchecked = len(checks) - len(checked)
    if mismatched:
        where = "; ".join(
            f"{check['arm']}: {' / '.join(check['problems'])}" for check in mismatched
        )
        status = (
            f"P2 어긋남 — 실험 결과가 아니라 버그입니다. 실험을 중단하고 이것을 먼저 고치세요 ({where})"
        )
    elif not checked:
        status = (
            f"P2 대조할 것 없음 — {unchecked}개 실행 모두 행을 떼어 두지 않았습니다 "
            "(예고와 실측이 만나는 지점이 없습니다)"
        )
    else:
        status = (
            f"P2 일치 — {len(checked)}개 실행에서 예고한 행 수가 실측과 같습니다"
            + (f" (대조 불가 {unchecked}개: 떼어 둔 행이 없음)" if unchecked else "")
        )
    return {
        "checks": checks,
        "checked": len(checked),
        "unchecked": unchecked,
        "mismatched": mismatched,
        "halt": bool(mismatched),
        "status": status,
    }


def p3(rows: list[Row]) -> dict[str, Any]:
    """P3: paired test deltas, marked observation only, never a verdict."""
    per_dataset = {
        row.dataset: {
            "delta": row.delta.delta,
            "ci_low": row.delta.ci_low,
            "ci_high": row.delta.ci_high,
            "excludes_zero": bool(row.delta.ci_low > 0 or row.delta.ci_high < 0),
            "identical_predictions": row.delta.identical_predictions,
        }
        for row in rows
        if row.delta is not None
    }
    excluding = sorted(name for name, entry in per_dataset.items() if entry["excludes_zero"])
    return {
        "observation_only": True,
        "per_dataset": per_dataset,
        "excluding_zero": excluding,
        "status": (
            f"P3 관찰 — Δ를 낸 데이터셋 {len(per_dataset)}개, 그중 CI가 0을 벗어난 것 "
            f"{len(excluding)}개{'(' + ', '.join(excluding) + ')' if excluding else ''}. "
            "판정으로 쓰지 않습니다 — 실행 간 잡음이 이 Δ에 섞여 있습니다 (REPEATS.md)"
        ),
    }


# --- Role: output -----------------------------------------------------------------------


def report(rows: list[Row]) -> None:
    for row in rows:
        print()
        if row.verdict != "ok":
            print(f"== {row.dataset} — 판정 거부")
            print(f"   {row.refusal}")
            continue
        print(
            f"== {row.dataset} ({row.metric}) · test {row.test_rows}행 · 지문 {row.test_fingerprint}"
        )
        auto = "켜짐" if row.forecast.get("early_stopping_auto_on") else "꺼짐"
        print(
            f"   예고: 학습 {row.forecast.get('train'):,}행 "
            f"(카드 {row.forecast.get('card_rows'):,}행) · early_stopping='auto' {auto} · "
            f"기본 fraction에서 {row.forecast.get('held_out_at_default_fraction'):,}행 떼어 냄"
        )
        keys = row.plan.get("keys") or {}
        for arm, _ in ARMS:
            named = keys.get(arm) or {}
            shown = " ".join(
                f"{key}={named.get(key)}" for key in PLAN_KEYS if named.get(key, ABSENT) is not ABSENT
            )
            model = (row.plan.get("model") or {}).get(arm)
            print(f"   {arm:<8} {str(model):<16} {shown or '(early stopping 관련 키를 안 적음)'}")
        if row.plan.get("differs"):
            print(f"   → 계획이 다릅니다: {', '.join(row.plan.get('differing_keys') or [])}")
        else:
            print("   → 계획이 같습니다")
        for check in row.p2:
            if not check.get("checked"):
                print(f"   P2 {check['arm']}: {check['reason']}")
            elif check.get("matches"):
                print(
                    f"   P2 {check['arm']}: 예고 {check['forecast_train']:,}행 = 실측 "
                    f"{check['measured_train']:,}행 (떼어 냄 {check['measured_held_out']:,})"
                )
            else:
                print(f"   P2 {check['arm']}: 어긋남 — {' / '.join(check['problems'])}")
        if row.delta is not None:
            ci = f"[{row.delta.ci_low:+.4f}, {row.delta.ci_high:+.4f}]"
            crosses = "" if (row.delta.ci_low > 0 or row.delta.ci_high < 0) else "  (0을 걸침)"
            print(f"   P3 Δ(rb_on − rb_off) {row.delta.delta:+.4f}  95% CI {ci}{crosses}")
        if row.caveat:
            print(f"   {row.caveat}")


def payload(rows: list[Row], resamples: int, seed: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/rowbudget.py",
        "preregistration": "docs/ROWBUDGET.md",
        "seed": seed,
        "iterations_per_run": FIRST_ITERATION,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "plan_keys": list(PLAN_KEYS),
        "threads": thread_state(),
        "datasets": [
            {
                "dataset": row.dataset,
                "metric": row.metric,
                "task": row.task,
                "verdict": row.verdict,
                "refusal": row.refusal,
                "test_rows": row.test_rows,
                "test_fingerprint": row.test_fingerprint,
                "missing_arms": row.missing_arms,
                "caveat": row.caveat,
                "forecast": row.forecast,
                "plan": row.plan,
                "p2": row.p2,
                "arms": row.arms,
                "delta": vars(row.delta) if row.delta is not None else None,
            }
            for row in rows
        ],
        "P1": p1(rows),
        "P2": p2(rows),
        "P3": p3(rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="row budget 프롬프트 절의 판정 (docs/ROWBUDGET.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help="이름 (기본: 전부)")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/rowbudget-seed<seed>.json"
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
        f"row budget 판정 · 시드 {args.seed} · iteration {FIRST_ITERATION}회 · "
        f"재추출 {args.resamples}회 · {describe_thread_state(thread_state())}"
    )
    legs: dict[str, dict[str, Scored]] = {}
    rows = []
    for name in names:
        collected: dict[str, Scored] = {}
        rows.append(adjudicate(BY_NAME[name], args.resamples, args.seed, collected))
        if collected:
            legs[name] = collected
    report(rows)

    out_path = args.out or OUT_DIR / f"rowbudget-seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(rows, args.resamples, args.seed), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {row.dataset: row.metric for row in rows})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print(p1(rows)["status"])
    verdict_p2 = p2(rows)
    print(verdict_p2["status"])
    print(p3(rows)["status"])
    print(f"판정 → {out_path.as_posix()}")
    refused = [row.dataset for row in rows if row.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
    if verdict_p2["halt"]:
        # Own exit code: runner and forecast disagree, a bug.
        print("P2가 어긋났으므로 P1·P3은 읽지 마세요.", file=sys.stderr)
        return 3
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
