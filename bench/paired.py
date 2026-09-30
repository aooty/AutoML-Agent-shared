"""Judge the 3-arm benchmark with a paired bootstrap of test differences.

Roles:

* Winners — find the model each arm picked.
* Rescoring — rebuild the split and score each winner again.
* Paired bootstrap — resample shared test rows for each delta.
* One dataset — score all arms, check splits, pair them.
* Preregistered criterion — check the success rule in RESULTS.md.
* Output — print the table, write JSON, CLI.

Usage:
    python -m bench.paired
    python -m bench.paired adult --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from automl_agent.config import MODEL_FILENAME, SCHEMA_FILENAME, read_json_object
from automl_agent.scoring.metrics import TASK_REGRESSION
from automl_agent.scoring.splits import split_three_way, val_fingerprint
from automl_agent.scripts.train import (
    LogBuffer,
    load_data,
    load_schema,
    scorers,
)
from automl_agent.scripts.train import (
    _proba as positive_proba,  # reused, not rewritten; see ``reproduce``
)
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import ALL, BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.random_search import RUNS_DIR
from bench.random_search import run_dir as random_run_dir

RESAMPLES = 4000
# logreg predictions wobble ~5.7e-08 across BLAS thread counts.
SCORE_TOLERANCE = 1e-6
ARTIFACTS_DIR = RUNS_DIR / "artifacts"
OUT_DIR = RUNS_DIR / "paired"


class Refusal(Exception):
    """Raised when a dataset cannot be judged, instead of a useless delta."""


# --- Role: winners ----------------------------------------------------------------------


@dataclass(frozen=True)
class Winner:
    """The model one arm picked, with its val and test scores."""

    arm: str
    label: str
    directory: Path
    config: dict[str, Any]
    val_score: float
    recorded_test: float
    fits: int


@dataclass
class Scored:
    """A winner scored again on the split this script rebuilt."""

    winner: Winner
    pred: Any
    y_test: Any
    test_score: float
    fingerprint: str
    task: str
    n_classes: int
    proba: Any = None
    average: str = "binary"

    @property
    def n_rows(self) -> int:
        return len(self.y_test)


def loop_winner(arm: str, directory: Path, metric: str) -> Winner | None:
    """Winner of a full loop run (``llm`` or ``no_llm``), read from ``history.json``.

    Only ``best.iteration`` there points to the kept model file."""
    history = read_json_object(directory / "history.json")
    if history is None:
        return None
    holdout = read_json_object(directory / "holdout.json")
    if holdout is None:
        raise Refusal(f"{arm}: history.json은 있는데 holdout.json이 없습니다 ({directory})")
    best = dict(history.get("best") or {})
    iteration = best.get("iteration")
    if not isinstance(iteration, int):
        raise Refusal(f"{arm}: history.json의 best.iteration이 정수가 아닙니다 ({iteration!r})")
    iter_dir = directory / "train" / f"iter_{iteration:02d}"
    recorded = dict(holdout.get("metrics") or {}).get(metric)
    if not isinstance(recorded, (int, float)):
        raise Refusal(f"{arm}: holdout.json에 {metric}이 없습니다 ({directory})")
    config = read_json_object(iter_dir / "train_config.json")
    if config is None:
        raise Refusal(f"{arm}: iteration {iteration}의 train_config.json을 읽을 수 없습니다")
    if not (iter_dir / MODEL_FILENAME).exists():
        raise Refusal(
            f"{arm}: iteration {iteration}의 {MODEL_FILENAME}이 없습니다 — "
            "--keep-models가 우승자까지 지웠거나 잘못된 iteration을 가리킵니다"
        )
    return Winner(
        arm=arm,
        label=f"iteration {iteration}",
        directory=iter_dir,
        config=config,
        val_score=float(best.get("score", float("nan"))),
        recorded_test=float(recorded),
        fits=int(history.get("iterations") or 0),
    )


def random_winner(directory: Path, k: int, metric: str) -> Winner:
    """Winner among the random arm's first k draws, read from ``summary.json``.

    The seed fixed draw order before any test, so no rerun is needed."""
    summary = read_json_object(directory / "summary.json")
    if summary is None:
        raise Refusal(f"random: summary.json을 읽을 수 없습니다 ({directory})")
    chosen = dict(summary.get("best_by_val_for_k") or {}).get(str(k))
    if not isinstance(chosen, int):
        raise Refusal(f"random: k={k}에 대한 best_by_val_for_k 항목이 없습니다")
    records = {int(r["draw"]): r for r in summary.get("records") or []}
    record = records.get(chosen)
    if record is None:
        raise Refusal(f"random: draw {chosen}의 기록이 summary.json에 없습니다")
    if str(summary.get("metric")) != metric:
        raise Refusal(
            f"random: summary.json의 지표가 {summary.get('metric')!r}인데 판정 지표는 {metric!r}입니다"
        )
    test = dict(record.get("test") or {})
    recorded = test.get("test_score")
    if not isinstance(recorded, (int, float)):
        raise Refusal(
            f"random: draw {chosen}에 test 점수가 없습니다 — "
            "앞 k회 우승자만 test를 봅니다, 이 뽑기가 그중 하나가 아닙니다"
        )
    draw_dir = Path(str(record["dir"]))
    config = read_json_object(draw_dir / "train_config.json")
    if config is None:
        raise Refusal(f"random: draw {chosen}의 train_config.json을 읽을 수 없습니다")
    if not (draw_dir / MODEL_FILENAME).exists():
        raise Refusal(f"random: draw {chosen}의 {MODEL_FILENAME}이 없습니다 ({draw_dir})")
    return Winner(
        arm=f"random@k={k}",
        label=f"draw {chosen}",
        directory=draw_dir,
        config=config,
        val_score=float(record.get("val_score", float("nan"))),
        recorded_test=float(recorded),
        fits=k,
    )


# --- Role: rescoring --------------------------------------------------------------------


def reproduce(winner: Winner, metric: str) -> Scored:
    """Score ``winner`` on test rows rebuilt from its own config.

    Uses the repo's own loader, split and scorers, so only the split can differ."""
    import joblib

    # load_data is chatty; keep its log but do not print.
    log = LogBuffer(echo=False)
    schema = load_schema(winner.directory / SCHEMA_FILENAME, log)
    x_arr, y_arr, n_classes, groups, task, _schema = load_data(winner.config, log, schema)
    seed = int(winner.config.get("seed", 42))
    splits = split_three_way(
        x_arr, y_arr, seed, groups=groups, stratify=task != TASK_REGRESSION
    )
    model = joblib.load(winner.directory / MODEL_FILENAME)
    y_test = np.asarray(splits.y_test)
    pred = model.predict(splits.x_test)
    proba = positive_proba(model, splits.x_test, n_classes, log)
    average = "binary" if n_classes == 2 else "macro"
    return Scored(
        winner=winner,
        pred=pred,
        y_test=y_test,
        proba=proba,
        test_score=score(metric, y_test, pred, proba, average, task),
        fingerprint=val_fingerprint(splits.x_test, y_test),
        task=task,
        n_classes=n_classes,
        average=average,
    )


def score(metric: str, y_true: Any, pred: Any, proba: Any, average: str, task: str) -> float:
    """``metric`` from the same scorer table the runs used."""
    thunk = scorers(y_true, pred, proba, average, task=task).get(metric)
    if thunk is None:
        raise Refusal(f"{metric}은 이 task({task})의 지표 목록에 없습니다")
    return float(thunk())


SPLIT_FIELDS = ("data", "target_missing", "seed", "task", "metric")


def split_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Config fields that decide which rows land in the test split."""
    return {key: config.get(key) for key in SPLIT_FIELDS if key in config}


# --- Role: paired bootstrap -------------------------------------------------------------


@dataclass
class Delta:
    a: str
    b: str
    kind: str
    delta: float
    ci_low: float
    ci_high: float
    p_better: float
    resamples_used: int
    identical_predictions: bool = False


def paired_delta(a: Scored, b: Scored, metric: str, resamples: int, seed: int, kind: str) -> Delta:
    """Bootstrap ``a - b`` over resamples of the shared test rows.

    Every pair reseeds the rng; resamples with one class are skipped and counted."""
    base = Delta(
        a=a.winner.arm, b=b.winner.arm, kind=kind, delta=a.test_score - b.test_score,
        ci_low=float("nan"), ci_high=float("nan"), p_better=float("nan"), resamples_used=0,
    )
    if np.array_equal(np.asarray(a.pred), np.asarray(b.pred)):
        base.identical_predictions = True
        return base
    y_test = a.y_test
    rng = np.random.default_rng(seed)
    n = len(y_test)
    deltas: list[float] = []
    for _ in range(resamples):
        rows = rng.integers(0, n, size=n)
        y_boot = y_test[rows]
        if a.task == TASK_REGRESSION:
            if float(np.var(y_boot)) == 0.0:
                continue
        elif len(np.unique(y_boot)) < 2:
            continue
        a_proba = None if a.proba is None else a.proba[rows]
        b_proba = None if b.proba is None else b.proba[rows]
        deltas.append(
            score(metric, y_boot, a.pred[rows], a_proba, a.average, a.task)
            - score(metric, y_boot, b.pred[rows], b_proba, b.average, b.task)
        )
    if not deltas:
        raise Refusal(f"{a.winner.arm} vs {b.winner.arm}: 쓸 수 있는 재추출이 없습니다")
    arr = np.asarray(deltas)
    base.ci_low, base.ci_high = (float(v) for v in np.quantile(arr, [0.025, 0.975]))
    base.p_better = float((arr > 0).mean())
    base.resamples_used = len(arr)
    return base


# --- Role: one dataset ------------------------------------------------------------------


@dataclass
class DatasetVerdict:
    dataset: str
    metric: str
    task: str
    verdict: str = "ok"
    refusal: str | None = None
    test_rows: int = 0
    test_fingerprint: str | None = None
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    pairs: list[Delta] = field(default_factory=list)
    missing_arms: list[str] = field(default_factory=list)


def adjudicate(
    dataset: Dataset, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    """_adjudicate | One dataset: fill ``out`` or raise Refusal."""
    metric = dataset.metric
    loops: dict[str, Winner] = {}
    for arm, prefix in (("llm", "llm"), ("no_llm", "nollm")):
        winner = loop_winner(arm, ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}", metric)
        if winner is None:
            out.missing_arms.append(arm)
        else:
            loops[arm] = winner

    random_dir = random_run_dir(dataset, seed)
    has_random = (random_dir / "summary.json").exists()
    if not has_random:
        out.missing_arms.append("random")
    if not loops and not has_random:
        raise Refusal(f"{dataset.name}: 시드 {seed}의 실행을 하나도 찾지 못했습니다")

    # Random prefixes needed: each loop arm's fit count, plus 5.
    winners: dict[str, Winner] = dict(loops)
    if has_random:
        needed = {5, *(w.fits for w in loops.values())}
        for k in sorted(needed):
            if k < 1:
                raise Refusal(f"{dataset.name}: 루프 팔의 학습 횟수가 {k}회로 기록됐습니다")
            winner = random_winner(random_dir, k, metric)
            winners[winner.arm] = winner

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
        raise Refusal(f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()}
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
            # Positive means validation made the winner look better.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
        }

    pairs: list[tuple[str, str, str]] = []
    for arm in ("llm", "no_llm"):
        if arm in scored and has_random:
            pairs.append((arm, f"random@k={scored[arm].winner.fits}", "primary"))
            if scored[arm].winner.fits != 5:
                pairs.append((arm, "random@k=5", "best_of_5"))
    if "llm" in scored and "no_llm" in scored:
        pairs.append(("llm", "no_llm", "primary"))

    for a_name, b_name, kind in pairs:
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, seed, kind)
        )

    # Last, so a refused dataset leaves no legs behind.
    if legs is not None:
        legs.update(scored)


# --- Role: preregistered criterion ------------------------------------------------------


def criterion(
    results: list[DatasetVerdict], challenger: str = "llm", seed: int = JUDGED_SEED
) -> dict[str, Any]:
    """Check the success rule committed to ``docs/RESULTS.md`` before any run.

    Binary datasets, primary pairs, judged seed only; partial coverage is called partial."""
    # Named list, so a new task type cannot grow the denominator.
    expected = set(JUDGED_NAMES)
    binary = [r for r in results if r.dataset in expected]
    wins, losses, ties, absent = [], [], [], []
    for result in binary:
        pair = next(
            (
                p for p in result.pairs
                if p.a == challenger and p.kind == "primary" and p.b.startswith("random")
            ),
            None,
        )
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        elif pair.ci_high < 0:
            losses.append(result.dataset)
        else:
            ties.append(result.dataset)
    missing = sorted(expected - {r.dataset for r in binary} - set(absent)) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if seed != JUDGED_SEED:
        complete = False
        status = (
            f"기준 판정 안 함 — 사전 등록은 시드 {JUDGED_SEED}만 판정에 씁니다. "
            f"시드 {seed}는 잡음 바닥용입니다 ({tally})"
        )
    elif missing:
        complete = False
        status = (
            f"부분 판정 — 기준은 이진 {len(expected)}개에 대한 것인데 "
            f"{len(wins) + len(ties) + len(losses)}개만 판정했습니다 "
            f"(없음: {', '.join(missing)}). 지금까지 {tally}"
        )
    else:
        complete = True
        if len(wins) >= 3:
            status = (
                f"기준 충족 — 이 {len(expected)}개 중 {len(wins)}개에서 "
                f"{challenger} 팔이 random search를 이겼습니다"
            )
        elif len(losses) >= 3:
            status = f"random search가 이겼습니다 — {len(losses)}개에서 CI가 0 아래입니다 ({tally})"
        else:
            status = f"구분되지 않음 — 이 데이터셋들과 이 예산으로는 답이 나오지 않습니다 ({tally})"
    return {
        "challenger": challenger,
        "seed": seed,
        "judged_seed": JUDGED_SEED,
        "expected_binary": sorted(expected),
        "judged": [r.dataset for r in binary if r.dataset not in absent],
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_evaluated": missing,
        "complete": complete,
        "status": status,
    }


# --- Role: output -----------------------------------------------------------------------


def report(results: list[DatasetVerdict]) -> None:
    for result in results:
        print()
        if result.verdict != "ok":
            print(f"== {result.dataset} — 판정 거부")
            print(f"   {result.refusal}")
            continue
        print(
            f"== {result.dataset} ({result.metric}) · test {result.test_rows}행 "
            f"· 지문 {result.test_fingerprint}"
        )
        if result.missing_arms:
            print(f"   없는 팔: {', '.join(result.missing_arms)}")
        print(f"   {'팔':<14} {'우승':<14} {'학습':>4} {'val':>9} {'test':>9} {'선택편향':>9}")
        for name, arm in result.arms.items():
            print(
                f"   {name:<14} {arm['label']:<14} {arm['fits']:>4} "
                f"{arm['val_score']:>9.4f} {arm['test_reproduced']:>9.4f} "
                f"{arm['selection_gap']:>+9.4f}"
            )
        if not result.pairs:
            print("   짝지은 비교 없음 (한 팔뿐)")
            continue
        print()
        print(f"   {'비교':<26} {'종류':<10} {'Δ':>9}  {'95% CI of Δ':<22} {'P(Δ>0)':>7} {'재추출':>7}")
        for pair in result.pairs:
            label = f"{pair.a} − {pair.b}"
            if pair.identical_predictions:
                print(
                    f"   {label:<26} {pair.kind:<10} {pair.delta:>+9.4f}  "
                    f"{'예측이 동일':<22} {'—':>7} {'—':>7}"
                )
                continue
            ci = f"[{pair.ci_low:+.4f}, {pair.ci_high:+.4f}]"
            crosses = "" if (pair.ci_low > 0 or pair.ci_high < 0) else "  (0을 걸침)"
            print(
                f"   {label:<26} {pair.kind:<10} {pair.delta:>+9.4f}  {ci:<22} "
                f"{pair.p_better:>7.3f} {pair.resamples_used:>7}{crosses}"
            )


def payload(results: list[DatasetVerdict], seed: int, resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/paired.py",
        "seed": seed,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # Thread state moves scores, so every verdict records it.
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
        "criterion": criterion(results, "llm", seed),
        "criterion_no_llm": criterion(results, "no_llm", seed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-arm 벤치마크의 짝지은 판정 (docs/RESULTS.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help="이름 (기본: 전부)")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path, default=None, help="기본: bench/runs/paired/seed<seed>.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Deferred: bench/predictions.py imports this module, a cycle.
    from bench.predictions import bundle_path, write_bundle

    args = parse_args(argv)
    names = args.datasets or [d.name for d in ALL]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"짝지은 판정 · 시드 {args.seed} · 재추출 {args.resamples}회 · "
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

    out_path = args.out or OUT_DIR / f"seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, args.seed, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {r.dataset: r.metric for r in results})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print(f"기준 (llm): {criterion(results, 'llm', args.seed)['status']}")
    print(f"기준 (no_llm): {criterion(results, 'no_llm', args.seed)['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
