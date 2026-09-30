"""Re-measure the RESULTS.md headline under the prompts used today.

Roles:

* Constants — datasets, arms, and the recorded headline numbers.
* Adjudication — score new and recorded arms, then pair them.
* Movement — what changed next to the recorded table.
* Output — the verdict file payload.
* CLI — parse flags, run, print, write files.

Usage:
    python -m bench.prompt_redo
    python -m bench.prompt_redo adult spambase
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
    Refusal,
    Scored,
    Winner,
    criterion,
    loop_winner,
    paired_delta,
    random_winner,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.random_search import run_dir as random_run_dir
from bench.replan import run_shape

# --- Role: constants ---------------------------------------------------------------------

# Four binary sets are judged; the other two only reported.
DATASETS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "speeddating",
    "spambase",
    "house_sales",
    "jungle-chess",
)

# The only seed all three arms share.
SEED = JUDGED_SEED

# Suffix 2 keeps the recorded RESULTS.md archives untouched.
ARMS: tuple[tuple[str, str], ...] = (("llm2", "llm2"), ("no_llm2", "nollm2"))

# Recorded RESULTS.md arms, rescored for free.
RECORDED_LOOP: tuple[str, str] = ("llm", "llm")
RECORDED_RULES: tuple[str, str] = ("no_llm", "nollm")

CHALLENGER = ARMS[0][0]
RULES = ARMS[1][0]

# Keys whose trainer handling 0ef38f8 changed.
CODE_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "early_stopping",
        "early_stopping_rounds",
        "scale_pos_weight",
        "validation_fraction",
        "n_iter_no_change",
    }
)

# Published headline deltas: llm minus budget-matched random.
RECORDED_PRIMARY: dict[str, float] = {
    "adult": 0.1599,
    "bank-marketing": 0.0860,
    "speeddating": 0.1267,
    "spambase": -0.0019,
    "house_sales": 0.3088,
    "jungle-chess": 0.2326,
}

# Published llm minus no_llm deltas; same split and cap.
RECORDED_VS_RULES: dict[str, float] = {
    "adult": 0.0046,
    "bank-marketing": 0.0913,
    "speeddating": 0.0652,
    "spambase": -0.0128,
    "house_sales": 0.0040,
    "jungle-chess": -0.0225,
}

# Recorded llm fits; more fits face a stronger random@k.
RECORDED_FITS: dict[str, int] = {
    "adult": 1,
    "bank-marketing": 1,
    "speeddating": 2,
    "spambase": 1,
    "house_sales": 1,
    "jungle-chess": 1,
}

# Binary datasets that made up the recorded 3 of 4.
RECORDED_WINS: tuple[str, ...] = ("adult", "bank-marketing", "speeddating")


# --- Role: adjudication ------------------------------------------------------------------


def _arm_legs(dataset: Dataset, out: DatasetVerdict) -> tuple[dict[str, Winner], dict[str, Any]]:
    """_arm_legs | Role: winners and run shapes of the two new arms."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{SEED}"
        winner = loop_winner(arm, directory, dataset.metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _recorded_legs(
    dataset: Dataset, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, Any]]:
    """_recorded_legs | Role: recorded RESULTS.md arms, read only."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in (RECORDED_LOOP, RECORDED_RULES):
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{SEED}"
        winner = loop_winner(arm, directory, dataset.metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def random_exposure(dataset: Dataset) -> dict[str, list[str]]:
    """Recorded random draws whose hyperparams touch :data:`CODE_SENSITIVE_KEYS`.

    Empty means the random arm is safe to reuse."""
    found: dict[str, list[str]] = {}
    directory = random_run_dir(dataset, SEED)
    for draw in sorted(directory.glob("draw_*")):
        config = read_json_object(draw / "train_config.json")
        if config is None:
            continue
        keys = sorted(
            key for key in dict(config.get("hyperparams") or {}) if key in CODE_SENSITIVE_KEYS
        )
        if keys:
            found[draw.name] = keys
    return found


def _pairs_to_take(scored: dict[str, Scored], has_random: bool) -> list[tuple[str, str, str]]:
    """_pairs_to_take | Role: comparisons, in the order the reused criterion reads."""
    pairs: list[tuple[str, str, str]] = []
    for arm in (CHALLENGER, RULES):
        if arm in scored and has_random:
            fits = scored[arm].winner.fits
            pairs.append((arm, f"random@k={fits}", "primary"))
            if fits != 5:
                pairs.append((arm, "random@k=5", "best_of_5"))
    if CHALLENGER in scored and RULES in scored:
        pairs.append((CHALLENGER, RULES, "primary"))
    if RULES in scored and RECORDED_RULES[0] in scored:
        # Rule arm calls no model, so only code moves it.
        pairs.append((RULES, RECORDED_RULES[0], "code_shift"))
    if CHALLENGER in scored and RECORDED_LOOP[0] in scored:
        # Prompt and code both differ here; observation only.
        pairs.append((CHALLENGER, RECORDED_LOOP[0], "prompt_shift"))
    return pairs


def _adjudicate(
    dataset: Dataset, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None = None
) -> None:
    """_adjudicate | Role: score all arms for one dataset; raises Refusal."""
    metric = dataset.metric
    winners, shapes = _arm_legs(dataset, out)
    if not winners:
        # The runs do not exist yet; not a data problem.
        raise Refusal(
            f"{dataset.name}: llm2도 no_llm2도 없습니다 — "
            "bench/scripts/run_prompt_redo.sh를 먼저 돌려야 판정할 수 있습니다"
        )
    recorded, recorded_shapes = _recorded_legs(dataset, out)
    winners.update(recorded)
    shapes.update(recorded_shapes)

    random_dir = random_run_dir(dataset, SEED)
    has_random = (random_dir / "summary.json").exists()
    if not has_random:
        out.missing_arms.append("random")
    else:
        # Each new arm's fit count, plus 5 for best_of_5.
        needed = {5, *(winners[arm].fits for arm, _ in ARMS if arm in winners)}
        for k in sorted(needed):
            if k < 1:
                raise Refusal(f"{dataset.name}: 새 팔의 학습 횟수가 {k}회로 기록됐습니다")
            leg = random_winner(random_dir, k, metric)
            winners[leg.arm] = leg

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
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
            f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다"
        )
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
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    for a_name, b_name, kind in _pairs_to_take(scored, has_random):
        out.pairs.append(paired_delta(scored[a_name], scored[b_name], metric, resamples, SEED, kind))

    # Last, so a refusal leaves no legs behind.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: Dataset, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Judge one dataset; a refusal clears the pairs."""
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --- Role: movement ----------------------------------------------------------------------


def _pair(result: DatasetVerdict, a: str, b_prefix: str, kind: str) -> Any:
    """_pair | Role: first pair matching arm, prefix and kind."""
    return next(
        (p for p in result.pairs if p.a == a and p.kind == kind and p.b.startswith(b_prefix)),
        None,
    )


def _bar(arms: dict[str, dict[str, Any]], *names: str) -> float | None:
    """_bar | Role: the auto bar from the first arm that recorded one."""
    for name in names:
        bar = arms.get(name, {}).get("goal_threshold")
        if bar is not None:
            return float(bar)
    return None


def _publish_pairs(result: DatasetVerdict) -> list[dict[str, Any]]:
    """_publish_pairs | Role: deltas as written, flagged when the bar changed."""
    rows: list[dict[str, Any]] = []
    for delta in result.pairs:
        row = dict(vars(delta))
        a_bar = result.arms.get(delta.a, {}).get("goal_threshold")
        b_bar = result.arms.get(delta.b, {}).get("goal_threshold")
        if a_bar is not None and b_bar is not None and a_bar != b_bar:
            row["spans_goal_change"] = True
            row["goal_threshold_a"] = a_bar
            row["goal_threshold_b"] = b_bar
        rows.append(row)
    return rows


def movement(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """Per dataset: headline deltas then and now, with fits and bar."""
    rows: list[dict[str, Any]] = []
    for result in results:
        primary = _pair(result, CHALLENGER, "random", "primary")
        vs_rules = _pair(result, CHALLENGER, RULES, "primary")
        # Loop arm first, so rule-only runs still read right.
        recorded_bar = _bar(result.arms, RECORDED_LOOP[0], RECORDED_RULES[0])
        now_bar = _bar(result.arms, CHALLENGER, RULES)
        rows.append(
            {
                "dataset": result.dataset,
                "judged": result.dataset in JUDGED_NAMES,
                "fits_recorded": RECORDED_FITS.get(result.dataset),
                "fits_now": result.arms.get(CHALLENGER, {}).get("fits"),
                "threshold_recorded": recorded_bar,
                "threshold_now": now_bar,
                "bar_moved": (
                    None
                    if recorded_bar is None or now_bar is None
                    else recorded_bar != now_bar
                ),
                "critic_runs_now": result.arms.get(CHALLENGER, {}).get("critic_runs"),
                "stop_reason_now": result.arms.get(CHALLENGER, {}).get("stop_reason"),
                "primary_recorded": RECORDED_PRIMARY.get(result.dataset),
                "primary_now": None if primary is None else primary.delta,
                "primary_ci": (
                    None if primary is None else [primary.ci_low, primary.ci_high]
                ),
                "vs_rules_recorded": RECORDED_VS_RULES.get(result.dataset),
                "vs_rules_now": None if vs_rules is None else vs_rules.delta,
                "vs_rules_ci": (
                    None if vs_rules is None else [vs_rules.ci_low, vs_rules.ci_high]
                ),
                "was_a_win": result.dataset in RECORDED_WINS,
            }
        )
    return rows


def core_question(results: list[DatasetVerdict]) -> dict[str, Any]:
    """Count ``llm2 − no_llm2`` wins on the four binary sets.

    A side reading from RESULTS.md, not the criterion."""
    wins, losses, ties, absent = [], [], [], []
    for result in results:
        if result.dataset not in JUDGED_NAMES:
            continue
        pair = _pair(result, CHALLENGER, RULES, "primary")
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif pair.identical_predictions or (pair.ci_low <= 0 <= pair.ci_high):
            ties.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        else:
            losses.append(result.dataset)
    missing = sorted(JUDGED_NAMES - {r.dataset for r in results}) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if missing:
        status = f"부분 — 이진 {len(JUDGED_NAMES)}개 중 일부가 없습니다 ({', '.join(missing)}). {tally}"
    elif len(wins) >= 3:
        status = f"루프 구조 위에 LLM이 더 얹습니다 — 이진 4개 중 {len(wins)}개 ({tally})"
    else:
        status = f"미달 — 3개 이상이 아닙니다 ({tally}). 기록된 표와 같은 방향입니다"
    return {
        "contrast": f"{CHALLENGER} − {RULES}",
        "note": "docs/RESULTS.md의 부 판정입니다. 사전 등록된 기준은 random 대비입니다",
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_evaluated": missing,
        "recorded": RECORDED_VS_RULES,
        "status": status,
    }


# --- Role: output ------------------------------------------------------------------------


def payload(results: list[DatasetVerdict], resamples: int) -> dict[str, Any]:
    """Build the verdict file contents."""
    return {
        "generated_by": "bench/prompt_redo.py",
        "preregistration": "docs/PROMPT-REDO.md",
        "remeasures": "docs/RESULTS.md",
        "seed": SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
        # Empty lists justify reusing the recorded random arm.
        "random_code_exposure": {
            r.dataset: random_exposure(BY_NAME[r.dataset]) for r in results
        },
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
                "pairs": _publish_pairs(r),
            }
            for r in results
        ],
        "movement": movement(results),
        # Reused from bench.paired, not rewritten.
        "criterion": criterion(results, CHALLENGER, SEED),
        "criterion_no_llm": criterion(results, RULES, SEED),
        "core_question": core_question(results),
    }


# --- Role: CLI ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "헤드라인 표를 지금 프롬프트로 다시 잰다 (docs/PROMPT-REDO.md의 사전 등록, "
            "기준은 docs/RESULTS.md의 것을 그대로 재사용)"
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help=f"이름 (기본: {' '.join(DATASETS)} — 그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help=f"기본: bench/runs/paired/prompt-redo-seed{SEED}.json"
    )
    return parser.parse_args(argv)


def _movement_text(recorded: float | None, now: float | None, ci: list[float] | None) -> str:
    """_movement_text | Role: format a recorded → now delta with its CI."""
    if recorded is None:
        return "—"
    if now is None:
        return f"{recorded:+.4f} → (없음)"
    span = "" if ci is None else f" [{ci[0]:+.4f}, {ci[1]:+.4f}]"
    return f"{recorded:+.4f} → {now:+.4f}{span}"


def _bar_text(row: dict[str, Any]) -> str:
    """_bar_text | Role: bar text, loud only when it moved."""
    recorded, now = row["threshold_recorded"], row["threshold_now"]
    if recorded is None or now is None:
        return "—"
    if not row["bar_moved"]:
        return f"{now:.4f}"
    return f"{recorded:.4f}→{now:.4f} !"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(
            f"사전 등록의 범위 밖 데이터셋: {', '.join(unknown)} (범위: {', '.join(DATASETS)})",
            file=sys.stderr,
        )
        return 2

    print(
        f"프롬프트 재측정 · 시드 {SEED} · 재추출 {args.resamples}회 · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: list[DatasetVerdict] = []
    for name in DATASETS:
        if name not in names:
            continue
        collected: dict[str, Scored] = {}
        results.append(adjudicate(BY_NAME[name], args.resamples, collected))
        if collected:
            legs[name] = collected
    report(results)

    out_path = args.out or OUT_DIR / f"prompt-redo-seed{SEED}.json"
    if not legs:
        # An empty verdict file would break a bare recheck.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_prompt_redo.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    # Built once so printed and written numbers match.
    body = payload(results, args.resamples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")

    print()
    print("기록된 표와 나란히 (기준은 이진 4개만, 나머지 둘은 따로 관찰):")
    print(
        f"   {'데이터셋':<16}{'학습':>9}{'바':>18} {'주 비교 (기록 → 지금)':<30} "
        f"{'llm − no_llm (기록 → 지금)':<28}"
    )
    for row in body["movement"]:
        fits = f"{row['fits_recorded']}→{row['fits_now']}"
        mark = "" if row["judged"] else " (따로)"
        primary = _movement_text(row["primary_recorded"], row["primary_now"], row["primary_ci"])
        rules = _movement_text(row["vs_rules_recorded"], row["vs_rules_now"], row["vs_rules_ci"])
        print(
            f"   {row['dataset'] + mark:<16}{fits:>9}{_bar_text(row):>18} "
            f"{primary:<30} {rules:<28}"
        )

    # A moved bar must be flagged before the numbers.
    moved = [row["dataset"] for row in body["movement"] if row["bar_moved"]]
    if moved:
        print()
        print(
            f"주의 — {', '.join(moved)}에서 auto 바가 12fa74b 이후로 올라갔습니다. "
            "이 카드의 code_shift·prompt_shift는 코드만이 아니라 목표까지 넘는 쌍이고, "
            "판정 파일이 그 쌍에 spans_goal_change를 답니다:"
        )
        for entry in body["datasets"]:
            for pair in entry["pairs"]:
                if pair.get("spans_goal_change"):
                    print(
                        f"   {entry['dataset']:<16}{pair['kind']:<13}"
                        f"{pair['a']} {pair['goal_threshold_a']:.4f} 대 "
                        f"{pair['b']} {pair['goal_threshold_b']:.4f}"
                    )

    exposure = {name: found for name, found in body["random_code_exposure"].items() if found}
    print()
    if exposure:
        print(
            "주의 — 아래 데이터셋의 random 뽑기가 0ef38f8이 손댄 키를 들고 있습니다. "
            "그 쌍은 코드가 섞인 비교로 인용해야 합니다:"
        )
        for name, found in exposure.items():
            print(f"   {name}: {found}")
    else:
        print(
            "random 뽑기 전부가 0ef38f8이 실행기에서 손댄 다섯 키를 하나도 들고 있지 않습니다 — "
            "그 팔을 다시 뽑지 않고 재사용하는 근거입니다. 12fa74b의 바 변경은 이 팔에 닿지 "
            "않습니다 (random_search.py는 목표를 읽지 않습니다)"
        )

    print()
    print(f"기준 (llm2): {body['criterion']['status']}")
    print(f"기준 (no_llm2): {body['criterion_no_llm']['status']}")
    print(f"핵심 질문 (llm2 − no_llm2): {body['core_question']['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
