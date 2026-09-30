"""Split the speeddating first-try gain into code, class_weight, and missing_count.

Roles:

* Constants — runs, leg names, keys, and the published gain.
* Refit prep — write the two configs a person could write.
* Legs — load legs, check the config ladder, pair them.
* Decomposition — the one-lever terms, their sum, and a summary.
* Output — JSON payload, prediction bundle, and the CLI.

Usage:
    python -m bench.class_weight_key --prepare
    python -m bench.class_weight_key
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED_SEED
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
    Refusal,
    Scored,
    Winner,
    paired_delta,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.replan import first_iteration_winner

# --- Role: constants ------------------------------------------------------------------

DATASET = "speeddating"
SEED = JUDGED_SEED

# Recorded runs from REPLAN.md and REPLAN-SEEDS.md; read only.
RECORDED_RUN = f"hardllm-{DATASET}-seed{SEED}"
NEW_RUN = f"hardllm2-{DATASET}-seed{SEED}"

# A third dir, so neither recorded archive is written.
REFIT_ROOT = ARTIFACTS_DIR / f"classweightkey-{DATASET}-seed{SEED}"

# Each ladder step A-C-D-B moves exactly one lever.
A = "recorded@it1"
C = "refit"
D = "refit+class_weight"
B = "new@it1"

# The measured key, and the flag that also changed.
KEY = "class_weight"
KEY_VALUE = "balanced"
FLAG = "missing_count"

# Published first-try gain being split, from REPLAN-SEEDS.md.
RECORDED_PLAN_SHIFT = 0.0480

REFITS: tuple[tuple[str, str], ...] = ((C, "refit"), (D, "refit-class-weight"))


# --- Role: refit prep -----------------------------------------------------------------


def refit_dir(name: str) -> Path:
    return REFIT_ROOT / dict(REFITS)[name]


def recorded_config() -> dict[str, Any]:
    path = ARTIFACTS_DIR / RECORDED_RUN / "train" / "iter_01" / "train_config.json"
    config = read_json_object(path)
    if config is None:
        raise Refusal(f"기록된 iteration 1의 train_config.json을 읽을 수 없습니다 ({path})")
    return config


def refit_configs() -> dict[str, dict[str, Any]]:
    """Derive C and D from A, adding at most one key."""
    base = recorded_config()
    with_key = copy.deepcopy(base)
    with_key.setdefault("hyperparams", {})[KEY] = KEY_VALUE
    return {C: copy.deepcopy(base), D: with_key}


def prepare() -> list[Path]:
    """Write the two refit configs and return their dirs."""
    written = []
    for name, config in refit_configs().items():
        directory = refit_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(directory)
    return written


# --- Role: legs -----------------------------------------------------------------------


def run_leg(name: str, arm: str, run: str, metric: str) -> Winner:
    """A run's iteration 1, renamed so pairs and bundle share one name."""
    winner = first_iteration_winner(arm, ARTIFACTS_DIR / run, metric)
    return replace(winner, arm=name)


def refit_winner(name: str, metric: str) -> Winner:
    """Wrap a refit as a ``Winner``; it has no recorded test score."""
    directory = refit_dir(name)
    result = read_json_object(directory / "result.json")
    if result is None:
        raise Refusal(
            f"{name}: result.json이 없습니다 ({directory}) — "
            "python -m bench.class_weight_key --prepare 뒤에 학습 두 번을 먼저 돌려야 합니다"
        )
    config = read_json_object(directory / "train_config.json")
    if config is None:
        raise Refusal(f"{name}: train_config.json을 읽을 수 없습니다 ({directory})")
    val = dict(result.get("metrics") or {}).get(metric)
    if not isinstance(val, (int, float)):
        raise Refusal(f"{name}: result.json에 {metric}이 없습니다 — 실패한 학습입니다")
    if not (directory / "model.joblib").exists():
        raise Refusal(f"{name}: model.joblib이 없습니다 ({directory})")
    return Winner(
        arm=name,
        label="refit",
        directory=directory,
        config=config,
        val_score=float(val),
        recorded_test=float("nan"),
        fits=1,
    )


def _describe(config: dict[str, Any]) -> dict[str, Any]:
    """_describe | Legs: the two fields the split depends on."""
    return {
        KEY: dict(config.get("hyperparams") or {}).get(KEY),
        FLAG: dict(config.get("preprocessing") or {}).get(FLAG),
    }


def _differing_paths(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """_differing_paths | Legs: dotted paths where two configs differ, recursing into dicts."""
    out: list[str] = []
    for key in sorted(set(left) | set(right)):
        lhs, rhs = left.get(key), right.get(key)
        if lhs == rhs:
            continue
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            out.extend(f"{key}.{sub}" for sub in _differing_paths(lhs, rhs))
        else:
            out.append(key)
    return out


def check_ladder(configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Refuse unless each neighbour pair differs in exactly its one field."""
    expected = {
        (A, C): [],
        (C, D): [f"hyperparams.{KEY}"],
        (D, B): [f"preprocessing.{FLAG}"],
    }
    found: dict[str, list[str]] = {}
    for (lhs, rhs), want in expected.items():
        paths = _differing_paths(configs[lhs], configs[rhs])
        found[f"{lhs} → {rhs}"] = paths
        if paths != want:
            raise Refusal(
                f"{lhs} → {rhs}: 달라야 하는 필드는 {want or '없음'}인데 실제로 다른 것은 "
                f"{paths or '없음'}입니다 — 한 걸음이 한 레버가 아니면 분해가 아닙니다"
            )
    if _describe(configs[B])[KEY] != KEY_VALUE:
        raise Refusal(
            f"{B}의 {KEY}가 {_describe(configs[B])[KEY]!r}입니다 — "
            f"이 모듈이 재는 것은 {KEY_VALUE!r}입니다"
        )
    return found


@dataclass
class Observation:
    """A verdict plus the config ladder that allowed it; kept off the shared dataclass."""

    verdict: DatasetVerdict
    ladder: dict[str, list[str]] = field(default_factory=dict)


def _adjudicate(
    resamples: int, out: DatasetVerdict, ladder: dict[str, list[str]], legs: dict[str, Scored] | None
) -> None:
    """_adjudicate | Legs: check the ladder, score the legs, and pair them."""
    dataset = BY_NAME[DATASET]
    metric = dataset.metric
    winners = {
        A: run_leg(A, "hard_llm", RECORDED_RUN, metric),
        C: refit_winner(C, metric),
        D: refit_winner(D, metric),
        B: run_leg(B, "hard_llm2", NEW_RUN, metric),
    }
    ladder.update(check_ladder({name: w.config for name, w in winners.items()}))

    scored = {name: reproduce(winner, metric) for name, winner in winners.items()}

    fingerprints = {name: s.fingerprint for name, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{name}={fp[:12]}" for name, fp in sorted(fingerprints.items()))
        raise Refusal(f"다리마다 test 지문이 다릅니다 ({listing}) — 같은 행이 아닙니다")
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(f"분할을 정하는 config 필드가 다리마다 다릅니다 ({sorted(set(identities.values()))})")

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
            **_describe(s.winner.config),
        }

    for a_name, b_name, kind in (
        (C, A, "code_shift"),
        (D, C, "key_shift"),
        (B, D, "count_shift"),
        (B, A, "plan_shift"),
    ):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, SEED, kind)
        )

    if legs is not None:
        legs.update(scored)


def adjudicate(resamples: int, legs: dict[str, Scored] | None = None) -> Observation:
    """Judge the four legs; a refusal empties the pairs."""
    dataset = BY_NAME[DATASET]
    out = DatasetVerdict(
        dataset=f"{DATASET}-seed{SEED}-it1", metric=dataset.metric, task=dataset.task
    )
    ladder: dict[str, list[str]] = {}
    try:
        _adjudicate(resamples, out, ladder, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return Observation(verdict=out, ladder=ladder)


# --- Role: decomposition --------------------------------------------------------------


def _delta(result: DatasetVerdict, kind: str) -> float:
    """_delta | Decomposition: the delta of one pair kind, or nan."""
    return next((p.delta for p in result.pairs if p.kind == kind), float("nan"))


def _val_agreement(result: DatasetVerdict) -> dict[str, Any]:
    """_val_agreement | Decomposition: does the refit match the recorded val score? Report only."""
    recorded = result.arms.get(A, {}).get("val_score")
    again = result.arms.get(C, {}).get("val_score")
    if not isinstance(recorded, (int, float)) or not isinstance(again, (int, float)):
        return {"recorded": recorded, "refit": again, "agrees": None}
    gap = abs(float(recorded) - float(again))
    return {
        "recorded": float(recorded),
        "refit": float(again),
        "gap": gap,
        "tolerance": SCORE_TOLERANCE,
        "agrees": gap <= SCORE_TOLERANCE,
    }


def decomposition(result: DatasetVerdict) -> dict[str, Any]:
    """Three one-lever terms, their sum, and the published total.

    A nonzero residual means a leg moved between pairs."""
    terms = {kind: _delta(result, kind) for kind in ("code_shift", "key_shift", "count_shift")}
    total = _delta(result, "plan_shift")
    summed = sum(terms.values())
    return {
        "terms": terms,
        "sum_of_terms": summed,
        "plan_shift": total,
        "residual": summed - total,
        "recorded_plan_shift": RECORDED_PLAN_SHIFT,
        "val_agreement": _val_agreement(result),
        "status": _status(result, terms, total),
    }


def _status(result: DatasetVerdict, terms: dict[str, float], total: float) -> str:
    """_status | Decomposition: the one-paragraph summary printed at the end."""
    if result.verdict != "ok":
        return f"판정 거부 — {result.refusal}"
    code, key, count = terms["code_shift"], terms["key_shift"], terms["count_shift"]
    identical = {
        p.kind for p in result.pairs if p.identical_predictions
    }
    parts = [
        f"기록된 첫 시도와 새 첫 시도의 차이 {total:+.4f}가 "
        f"코드 {code:+.4f} · {KEY} {key:+.4f} · {FLAG} {count:+.4f}로 갈립니다"
    ]
    if "code_shift" in identical:
        parts.append("코드 항은 예측이 동일해 0입니다 (같은 config, 두 코드 버전)")
    else:
        parts.append(
            f"코드 항이 0이 아닙니다 ({code:+.4f}) — 같은 config가 코드 버전에 따라 다르게 "
            "예측했으므로 나머지 두 항도 그만큼 순수하지 않습니다"
        )
    parts.append(
        f"REPLAN-SEEDS.md가 {RECORDED_PLAN_SHIFT:+.4f}를 {KEY} 한 키에 귀속했는데, "
        f"그 몫은 {key:+.4f}이고 {FLAG}가 {count:+.4f}를 따로 가져갔습니다"
    )
    return ". ".join(parts)


# --- Role: output ---------------------------------------------------------------------


def payload(observation: Observation, resamples: int) -> dict[str, Any]:
    result = observation.verdict
    return {
        "generated_by": "bench/class_weight_key.py",
        "corrects": "docs/REPLAN-SEEDS.md",
        "dataset": DATASET,
        "run_seed": SEED,
        "seed": SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
        "preregistered": False,
        "datasets": [
            {
                "dataset": result.dataset,
                "metric": result.metric,
                "task": result.task,
                "verdict": result.verdict,
                "refusal": result.refusal,
                "test_rows": result.test_rows,
                "test_fingerprint": result.test_fingerprint,
                "config_ladder": observation.ladder,
                "arms": result.arms,
                "pairs": [vars(p) for p in result.pairs],
            }
        ],
        "decomposition": decomposition(result),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "speeddating 첫 시도의 +0.0480을 코드 · class_weight · missing_count로 갈라 봅니다 "
            "(사전 등록 없는 관찰, docs/REPLAN-SEEDS.md의 덧붙임)"
        )
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="재학습할 두 config를 쓰고 학습 명령을 찍습니다 (판정하지 않습니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"기본: bench/runs/paired/class-weight-key-{DATASET}.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.prepare:
        try:
            written = prepare()
        except Refusal as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("재학습할 config를 썼습니다. 학습 두 번을 돌린 뒤 이 모듈을 인수 없이 다시 부르세요:")
        for directory in written:
            print(
                f"   python -m automl_agent.scripts.train "
                f"--config {(directory / 'train_config.json').as_posix()} "
                f"--out {(directory / 'result.json').as_posix()}"
            )
        print("   (OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 로 돌리세요 — "
              "기록된 다리가 그 설정입니다)")
        return 0

    print(
        f"첫 시도의 차이를 갈라 보기 · {DATASET} 시드 {SEED} · 재추출 {args.resamples}회 "
        f"· {describe_thread_state(thread_state())}"
    )
    legs: dict[str, dict[str, Scored]] = {}
    collected: dict[str, Scored] = {}
    observation = adjudicate(args.resamples, collected)
    result = observation.verdict
    if collected:
        legs[result.dataset] = collected
    report([result])

    out_path = args.out or OUT_DIR / f"class-weight-key-{DATASET}.json"
    if not legs:
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "python -m bench.class_weight_key --prepare 를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(observation, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {result.dataset: result.metric})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    split = decomposition(result)
    print("분해:")
    for kind, value in split["terms"].items():
        print(f"   {kind:<12} {value:+.4f}")
    print(f"   {'합계':<12} {split['sum_of_terms']:+.4f}   (plan_shift {split['plan_shift']:+.4f}, "
          f"잔차 {split['residual']:+.2e})")
    agreement = split["val_agreement"]
    if agreement.get("agrees") is not None:
        verdict = "일치" if agreement["agrees"] else f"불일치 ({agreement['gap']:.2e})"
        print(f"   재학습 val {agreement['refit']:.6f} 대 기록 {agreement['recorded']:.6f} — {verdict}")
    print()
    print(split["status"])
    print(f"판정 → {out_path.as_posix()}")
    if result.verdict != "ok":
        return 1
    if not math.isclose(split["residual"], 0.0, abs_tol=SCORE_TOLERANCE):
        print(
            f"분해의 잔차가 {split['residual']:+.2e}입니다 — 다리가 쌍 사이에서 움직였습니다",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
