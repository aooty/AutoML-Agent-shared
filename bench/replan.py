"""Judge the replan arms: does the loop turn budget into score?

Roles:

* Constants — the two arms and the fixed dataset scope.
* Run legs — run shape and the iteration 1 leg.
* Adjudication — score every leg and pair them per dataset.
* Criterion — judge the rule fixed in docs/REPLAN.md.
* Output — budget table, JSON payload, and the CLI.

Usage:
    python -m bench.replan
    python -m bench.replan adult --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import MODEL_FILENAME, read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED_SEED, Dataset
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
    random_winner,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.random_search import run_dir as random_run_dir

# --- Role: constants ------------------------------------------------------------------

# Both run at the raised bar; hard_nollm replans by rules.
ARMS: tuple[tuple[str, str], ...] = (("hard_llm", "hardllm"), ("hard_nollm", "hardnollm"))

# Binary only; bank-marketing's bar does not move. Fixed here.
SCOPE: tuple[str, ...] = ("adult", "spambase", "speeddating")


# --- Role: run legs -------------------------------------------------------------------


def run_shape(directory: Path) -> dict[str, Any]:
    """How the loop ran: stop reason, critic runs, and the bar faced.

    Fits live in ``Winner.fits``; critic runs are counted from the attempts."""
    history = read_json_object(directory / "history.json") or {}
    attempts = history.get("history") or []
    return {
        "stop_reason": history.get("stop_reason"),
        "critic_runs": sum(1 for a in attempts if isinstance(a, dict) and a.get("critic")),
        "goal_threshold": dict(history.get("goal") or {}).get("threshold"),
    }


def first_iteration_winner(arm: str, directory: Path, metric: str) -> Winner:
    """Wrap a run's iteration 1 as a ``Winner`` for pairing.

    ``recorded_test`` is nan: holdout only ever scores the final winner."""
    history = read_json_object(directory / "history.json")
    if history is None:
        raise Refusal(f"{arm}: history.json을 읽을 수 없습니다 ({directory})")
    attempts = history.get("history") or []
    if not attempts:
        raise Refusal(f"{arm}: history가 비어 있습니다 ({directory})")
    first = attempts[0]
    val = dict(first.get("metrics") or {}).get(metric)
    if not isinstance(val, (int, float)):
        raise Refusal(f"{arm}: iteration 1의 {metric}이 history에 없습니다 — 실패한 시도입니다")
    iter_dir = directory / "train" / "iter_01"
    config = read_json_object(iter_dir / "train_config.json")
    if config is None:
        raise Refusal(f"{arm}: iteration 1의 train_config.json을 읽을 수 없습니다 ({iter_dir})")
    if not (iter_dir / MODEL_FILENAME).exists():
        raise Refusal(
            f"{arm}: iteration 1의 {MODEL_FILENAME}이 없습니다 — 이 실험은 "
            "--keep-models all 로 돌려야 합니다 (기본값은 우승자 아닌 iteration을 지웁니다)"
        )
    return Winner(
        arm=f"{arm}@it1",
        label="iteration 1",
        directory=iter_dir,
        config=config,
        val_score=float(val),
        recorded_test=float("nan"),
        fits=1,
    )


# --- Role: adjudication ---------------------------------------------------------------


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    """_adjudicate | Adjudication: score legs, check same rows, and fill ``out``."""
    metric = dataset.metric
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        winners[f"{arm}@it1"] = first_iteration_winner(arm, directory, metric)
        shapes[arm] = run_shape(directory)

    if "hard_llm" not in winners:
        # Not a data problem: the run has not happened yet.
        raise Refusal(
            f"{dataset.name}: 시드 {seed}의 hard_llm 실행이 없습니다 — "
            "bench/scripts/run_replan.sh를 먼저 돌려야 판정할 수 있습니다"
        )

    # Random legs at k = LLM fits and at k=1.
    random_dir = random_run_dir(dataset, seed)
    if (random_dir / "summary.json").exists():
        for k in sorted({1, winners["hard_llm"].fits}):
            if k < 1:
                raise Refusal(f"{dataset.name}: hard_llm의 학습 횟수가 {k}회로 기록됐습니다")
            leg = random_winner(random_dir, k, metric)
            winners[leg.arm] = leg
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # Iteration 1 legs never had a test score (nan).
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
        raise Refusal(f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset.name}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )

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
            # Cost of picking the best val; key here, not side.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    matched = f"random@k={winners['hard_llm'].fits}"
    pairs: list[tuple[str, str, str]] = [("hard_llm", "hard_llm@it1", "primary")]
    if matched in scored and "random@k=1" in scored:
        # Always computed: must sit beside any positive primary.
        pairs.append((matched, "random@k=1", "blind_gain"))
    if matched in scored:
        pairs.append(("hard_llm", matched, "matched_level"))
    if "hard_nollm" in scored:
        pairs.append(("hard_llm", "hard_nollm", "vs_rules"))
        pairs.append(("hard_nollm", "hard_nollm@it1", "rules_gain"))

    for a_name, b_name, kind in pairs:
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, seed, kind)
        )

    # Last, so a refusal leaves no legs in the bundle.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: Dataset, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Judge one dataset; a refusal empties the pairs."""
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --- Role: criterion ------------------------------------------------------------------


def _primary(result: DatasetVerdict) -> Any:
    """_primary | Criterion: the primary pair of one dataset, or None."""
    return next(
        (
            p
            for p in result.pairs
            if p.a == "hard_llm" and p.b == "hard_llm@it1" and p.kind == "primary"
        ),
        None,
    )


def criterion(results: list[DatasetVerdict], seed: int = JUDGED_SEED) -> dict[str, Any]:
    """Judge the rule committed in docs/REPLAN.md before these runs.

    A majority of looped datasets must win; a losing majority is reported too."""
    in_scope = [r for r in results if r.dataset in SCOPE]
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    not_looped: list[str] = []
    absent: list[str] = []
    for result in in_scope:
        pair = _primary(result)
        critic_runs = result.arms.get("hard_llm", {}).get("critic_runs")
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif not isinstance(critic_runs, int) or critic_runs < 1:
            not_looped.append(result.dataset)
        elif pair.identical_predictions:
            ties.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        elif pair.ci_high < 0:
            losses.append(result.dataset)
        else:
            ties.append(result.dataset)

    judged = wins + ties + losses
    missing = sorted(set(SCOPE) - {r.dataset for r in in_scope}) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if not_looped:
        tally += f" / 루프 안 돔 {len(not_looped)}"
    needed = len(judged) // 2 + 1

    if seed != JUDGED_SEED:
        complete = False
        status = f"기준 판정 안 함 — 사전 등록은 시드 {JUDGED_SEED}만 판정에 씁니다 ({tally})"
    elif missing:
        complete = False
        status = (
            f"부분 판정 — 기준의 범위는 {len(SCOPE)}개인데 판정할 수 있는 것이 "
            f"{len(judged)}개입니다 (없음: {', '.join(missing)}). 지금까지 {tally}"
        )
    elif len(judged) < 2:
        complete = False
        status = (
            f"판정 불가 — 루프가 돈 데이터셋이 {len(judged)}개뿐입니다 "
            f"(안 돈 것: {', '.join(not_looped) or '없음'}). 바를 올려도 미달이 되지 않았다면 "
            "이 실험의 전제가 깨진 것이고, 기준을 약하게 고치지 않습니다"
        )
    else:
        complete = True
        if len(wins) >= needed:
            status = (
                f"기준 충족 — 루프가 돈 {len(judged)}개 중 {len(wins)}개에서 재계획이 자기 "
                "iteration 1을 이겼습니다. random의 같은 예산 이득(blind_gain)과 hard_nollm "
                "시드 산포를 반드시 함께 인용합니다"
            )
        elif len(losses) >= needed:
            status = f"재계획이 점수를 내립니다 — {len(losses)}개에서 CI가 0 아래입니다 ({tally})"
        else:
            status = f"구분되지 않음 — 이 데이터셋들과 이 예산으로는 답이 나오지 않습니다 ({tally})"
    return {
        "challenger": "hard_llm",
        "baseline": "hard_llm@it1",
        "seed": seed,
        "judged_seed": JUDGED_SEED,
        "scope": list(SCOPE),
        "judged": judged,
        "majority_needed": needed,
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_looped": not_looped,
        "not_evaluated": missing,
        "complete": complete,
        "status": status,
    }


# --- Role: output ---------------------------------------------------------------------


def budget_used(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """Per dataset, what each arm spent: fits, stop reason, critic runs, bar."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"dataset": result.dataset}
        for arm, _ in ARMS:
            arm_row = result.arms.get(arm, {})
            row[arm] = {
                "fits": arm_row.get("fits"),
                "stop_reason": arm_row.get("stop_reason"),
                "critic_runs": arm_row.get("critic_runs"),
                "goal_threshold": arm_row.get("goal_threshold"),
                "winner": arm_row.get("label"),
            }
        rows.append(row)
    return rows


def payload(results: list[DatasetVerdict], seed: int, resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/replan.py",
        "preregistration": "docs/REPLAN.md",
        "margin": 0.5,
        "seed": seed,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # Same recorder as the runs, including cpu_count.
        "threads": thread_state(),
        "datasets": [
            {
                "dataset": r.dataset,
                "metric": r.metric,
                "task": r.task,
                "verdict": r.verdict,
                "refusal": r.refusal,
                "test_rows": r.test_rows,
                "test_fingerprint": r.test_fingerprint,
                "missing_arms": r.missing_arms,
                "arms": r.arms,
                "pairs": [vars(p) for p in r.pairs],
            }
            for r in results
        ],
        "budget": budget_used(results),
        "criterion": criterion(results, seed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="미달인 바에서 재계획이 예산을 점수로 바꾸는가 (docs/REPLAN.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help=f"이름 (기본: {', '.join(SCOPE)})")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/replan-seed<seed>.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or list(SCOPE)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"재계획 판정 · margin 0.5 · 시드 {args.seed} · 재추출 {args.resamples}회 · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results = []
    for name in names:
        collected: dict[str, Scored] = {}
        results.append(adjudicate(BY_NAME[name], args.seed, args.resamples, collected))
        if collected:
            legs[name] = collected
    report(results)

    out_path = args.out or OUT_DIR / f"replan-seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, args.seed, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Models are never committed; the bundle lets recheck redo deltas.
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {r.dataset: r.metric for r in results})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print("예산 — 실제로 쓴 것 (정체 가드가 미달인 바에서도 먼저 걸릴 수 있습니다):")
    for row in budget_used(results):
        for arm, _ in ARMS:
            spent = row[arm]
            print(
                f"   {row['dataset']:<14} {arm:<11} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"바={spent['goal_threshold']} 우승={spent['winner']}"
            )
    print()
    print(f"기준: {criterion(results, args.seed)['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
