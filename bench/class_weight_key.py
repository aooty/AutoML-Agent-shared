"""새 프롬프트가 speeddating의 *첫 시도*에 대해 실제로 무엇을 바꿨는가?

``docs/REPLAN-SEEDS.md``는 새 프롬프트가 iteration 1의 test 점수를 0.6895에서 0.7375로
— **+0.0480** — 올렸고, 주 Δ가 +0.0935에서 +0.0507로 내려간 것은 재계획 능력을 잃어서가 아니라
이 상승 때문이라고 적었다. 그리고 그 +0.0480을 하이퍼파라미터 하나(``class_weight: "balanced"``)에
귀속하며 두 계획이 그 키 하나로만 다르다고 했다.

**그 문장은 틀렸고, 이 모듈은 그것을 적어 둔 덕에 확인할 수 있게 됐기 때문에 존재한다.** 두
``train_config.json``은 *두* 곳에서 다르다:

* ``hyperparams.class_weight`` — 기록된 계획에는 없고, 새 계획에는 ``"balanced"``;
* ``preprocessing.missing_count`` — 기록된 계획은 ``true``, 새 계획은 ``false``.

한 줄에 레버 둘은 이 저장소의 원장이 해석을 거부해 온 바로 그 모양이므로, 두 config 사이의 짝지은
Δ를 "키 하나의 효과"라고 부를 수 없다 — ``REPLAN-SEEDS.md``의 덧붙임이 지금 그렇게 다시 적어 둔
내용이다.

가르는 값은 CPU 적합 두 번이고 API 호출은 없다. 중간 config 둘이 모두 사람이 적어 낼 수 있는
것이기 때문이다:

===============================  ======================  ===================  =================
다리                               코드                      ``class_weight``     ``missing_count``
===============================  ======================  ===================  =================
``recorded@it1``  (A)            기록된 실행의 것               없음                   ``true``
``refit``         (C)            현재                      없음                   ``true``
``refit+class_weight`` (D)       현재                      ``"balanced"``       ``true``
``new@it1``       (B)            현재                      ``"balanced"``       ``false``
===============================  ======================  ===================  =================

이웃한 쌍마다 정확히 한 가지만 움직이므로, 다리 넷은 발표된 +0.0480을 정확히 그 합이 되는 세 항으로
분해한다 (같은 행에 대고 낸 점수의 차이들이다):

* ``code_shift`` = C − A — config를 고정한 채 코드만. 0일 것으로 예상한다. ``REPLAN-SEEDS``가 이미
  규칙 팔의 ``code_shift``를 세 시드 모두에서 동일한 예측으로 측정했다.
* ``key_shift`` = D − C — 코드 버전 하나 아래에서 ``class_weight`` 하나만. ``REPLAN-SEEDS.md``가
  가졌다고 주장했으나 실은 갖지 않았던 수다.
* ``count_shift`` = B − D — ``missing_count`` 하나만. 이름을 따로 줄 값이 있다: ``capabilities.py``가
  이제 ``missing_count``가 MIMIC에서 아무것도 사지 못했다는 노트를 담고 있고, 그 노트는 계획
  프롬프트에 렌더되므로, "계획이 그 키를 더는 적지 않았다"는 프롬프트 수정이 제 일을 한 것일 수 있다.
* ``plan_shift`` = B − A — 발표된 +0.0480. 분해를 그 분해 대상과 대조할 수 있게 남겨 둔다.

**사전 등록도 없고 성공 기준도 없다.** 통과/실패가 없는 관찰이고, 모든 값은 나온 그대로 적는다.
대신 가드가 하나 있다 — 점수를 내기 전에 세 config를 필드 단위로 비교하고, 어긋나면 발표하지 않고
거부한다. 두 키가 다른 config 사이에서 잰 "단일 키 효과"가 바로 이 모듈이 정정하려고 쓰인 그
실수이기 때문이다.

말할 수 없는 것 셋, 덧붙임에도 같이 적었다:

* 데이터셋 하나, 시드 하나, 지표 하나;
* *첫 시도*만 — 여기 있는 어느 것도 그 뒤 재계획이 하는 일에 대한 것이 아니다;
* "그 시드의 첫 계획이 그 키를 썼다"이고, "프롬프트 문장이 이 이득을 일으킨다"가 아니다. 프롬프트
  수정은 커밋 하나이고, 한 config에서 키가 내는 효과는 그 문장이 계획에 내는 효과가 아니다.

사용법:
    python -m bench.class_weight_key --prepare   # 재학습할 config 둘을 쓰고 명령을 찍는다
    python -m bench.class_weight_key             # 재학습 둘이 존재하면 판정한다
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

DATASET = "speeddating"
SEED = JUDGED_SEED

# 기록된 실행 둘. ``hardllm-``은 ``REPLAN.md``의 아카이브, ``hardllm2-``는 ``REPLAN-SEEDS.md``의
# 것이다. 이 모듈은 이들을 읽기만 한다.
RECORDED_RUN = f"hardllm-{DATASET}-seed{SEED}"
NEW_RUN = f"hardllm2-{DATASET}-seed{SEED}"

# 재학습 둘이 갈 자리. 세 번째 디렉터리인 이유는 기록된 아카이브 어느 쪽에도 쓰지 않기 위해서다.
REFIT_ROOT = ARTIFACTS_DIR / f"classweightkey-{DATASET}-seed{SEED}"

A = "recorded@it1"
C = "refit"
D = "refit+class_weight"
B = "new@it1"

# 이 모듈 이름의 유래인 하이퍼파라미터 하나와, 그 옆에 앉아 있던 것으로 드러난 preprocessing
# 플래그. 둘을 여기 적어 두어 가드가 config에 우연히 들어 있는 값이 아니라 상수와 견주게 한다.
KEY = "class_weight"
KEY_VALUE = "balanced"
FLAG = "missing_count"

# 분해 대상인 발표된 수. ``REPLAN-SEEDS.md``의 "예측과 맞춰 보기" 표에서 왔다.
RECORDED_PLAN_SHIFT = 0.0480

REFITS: tuple[tuple[str, str], ...] = ((C, "refit"), (D, "refit-class-weight"))


def refit_dir(name: str) -> Path:
    return REFIT_ROOT / dict(REFITS)[name]


# --------------------------------------------------------------------------- #
# 사람이 적어 낼 수 있는 config 둘 준비하기
# --------------------------------------------------------------------------- #


def recorded_config() -> dict[str, Any]:
    path = ARTIFACTS_DIR / RECORDED_RUN / "train" / "iter_01" / "train_config.json"
    config = read_json_object(path)
    if config is None:
        raise Refusal(f"기록된 iteration 1의 train_config.json을 읽을 수 없습니다 ({path})")
    return config


def refit_configs() -> dict[str, dict[str, Any]]:
    """A에 키를 최대 하나 더해 얻은 C와 D.

    손으로 적지 않고 파생시킨다 — 손으로 베낀 config는 어긋날 수 있는 다섯 번째 물건이고, 이 모듈이
    들여도 되는 차이는 재고 있는 그 하나뿐이다.
    """
    base = recorded_config()
    with_key = copy.deepcopy(base)
    with_key.setdefault("hyperparams", {})[KEY] = KEY_VALUE
    return {C: copy.deepcopy(base), D: with_key}


def prepare() -> list[Path]:
    """재학습할 config 둘을 쓰고 그 디렉터리를 돌려준다."""
    written = []
    for name, config in refit_configs().items():
        directory = refit_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(directory)
    return written


# --------------------------------------------------------------------------- #
# 다리들
# --------------------------------------------------------------------------- #


def run_leg(name: str, arm: str, run: str, metric: str) -> Winner:
    """기록된 실행의 iteration 1을, 이 모듈의 다리 이름으로 바꿔 부른 것.

    이름 바꾸기는 장식이 아니다. ``paired_delta``는 쌍에 ``Winner.arm``으로 라벨을 붙이고
    ``bench/predictions.py``는 번들을 dict 키로 색인한다 — 그래서 ``arm``은 ``hard_llm@it1``인데
    번들은 ``recorded@it1``이라고 적힌 다리는, ``python -m bench.recheck``가 저장된 어느 예측과도
    맞춰 볼 수 없는 Δ를 발표한다. 다리마다 이름 하나.
    """
    winner = first_iteration_winner(arm, ARTIFACTS_DIR / run, metric)
    return replace(winner, arm=name)


def refit_winner(name: str, metric: str) -> Winner:
    """재학습을 ``Winner``로 감싼 것. 실행 다리들과 같은 짝짓기 장치를 통과하게 한다.

    ``recorded_test``가 ``nan``인 이유는 iteration 1 다리들과 같다 — 여기 있는 어느 것도 test 분할을
    본 적이 없으므로, 다시 낸 값을 견줄 기록된 test 점수가 없다. 실제로 적용되는 확인은
    ``_adjudicate``가 모든 다리에 돌리는 것들이다: config의 분할 필드와 test 행의 지문.
    """
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
    """분해가 걸려 있는 필드 둘을, config에 앉아 있는 모양 그대로."""
    return {
        KEY: dict(config.get("hyperparams") or {}).get(KEY),
        FLAG: dict(config.get("preprocessing") or {}).get(FLAG),
    }


def _differing_paths(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """두 config가 어긋나는 모든 ``a.b`` 경로. 중첩 객체는 한 단계까지 들어간다."""
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
    """이웃한 config 쌍마다 달라야 하는 그 한 필드만 다르지 않으면 거부한다.

    이것이 가드 전부이고, 이 모듈이 있는 이유다. ``REPLAN-SEEDS.md``는 두 키가 다른 config 쌍에 대해
    단일 키 귀속을 발표했다. 단이 한 걸음씩 떨어져 있지 않은 사다리는 분해할 수 없으므로, 유보를 달아
    보고하지 않고 거부한다 — 유보는 그 틀린 문장이 이미 달고 있었다.
    """
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
    """판정과, 그 판정을 허락한 사다리.

    ``DatasetVerdict``에 필드를 더하지 않고 따로 담는 이유는, config 사다리가 이 모듈만의
    선행조건이기 때문이다. 공용 dataclass에 얹으면 ``bench/runs/paired/``의 다른 모든 판정 파일이 그
    키를 null로 달게 된다.
    """

    verdict: DatasetVerdict
    ladder: dict[str, list[str]] = field(default_factory=dict)


def _adjudicate(
    resamples: int, out: DatasetVerdict, ladder: dict[str, list[str]], legs: dict[str, Scored] | None
) -> None:
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


# --------------------------------------------------------------------------- #
# 다리 넷이 말하는 것
# --------------------------------------------------------------------------- #


def _delta(result: DatasetVerdict, kind: str) -> float:
    return next((p.delta for p in result.pairs if p.kind == kind), float("nan"))


def _val_agreement(result: DatasetVerdict) -> dict[str, Any]:
    """기록된 config를 현재 코드로 다시 학습하면 기록된 *val* 점수가 나오는가?

    코드 항을 두 번째로, 독립적으로 보는 것이다. 짝지은 Δ는 이 모듈이 스스로 낸 test 행에 대고
    계산하는데, 이쪽은 이 모듈이 계산하지 않은 두 수를 견준다 — 기록된 실행이 적어 둔 val 점수와
    재학습이 적어 둔 val 점수. 거부하지 않고 적기만 한다. 여기서 어긋나는 것은 멈출 이유가 아니라
    발견이다.
    """
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
    """한 레버짜리 항 셋, 그 합, 그리고 합이 되어야 하는 발표된 수.

    잔차는 증거가 아니라 산술이다 — 같은 행에 대고 낸 점수의 차이 셋은 구성상 네 번째와 같아지므로,
    0이 아닌 잔차는 다리가 쌍 사이에서 움직였다는 뜻이다. 그래도 발표한다. "구성상 0이어야 한다"는
    것이 바로 파일에서 확인할 수 있어야 하는 종류의 주장이기 때문이다.
    """
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
