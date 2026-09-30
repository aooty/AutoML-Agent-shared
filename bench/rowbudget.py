"""``docs/ROWBUDGET.md``의 판정: row budget 프롬프트 절이 계획을 바꾸는가?

사전 등록은 ``docs/ROWBUDGET.md``이고, **이 파일은 그 실행들이 존재하기 전에 커밋됐다** —
``paired.py``·``spg.py``·``repeats.py``와 같은 규칙, 같은 이유다.

이 실험이 다른 점: **관측 대상이 점수가 아니라 계획이다.** 이 절이 있는 이유는 계획이 잘못 추론한
행 수에 대고 용량과 early stopping을 정하는 것을 막는 것이므로, 답할 수 있는 질문은 iteration 1의
계획이 달라지는가다. 실행이 ``--max-iterations 1``인 이유는 iteration 2부터는 결과를 읽어서 이 절의
효과와 "다른 결과를 읽었다"를 섞기 때문이다. P3은 짝지은 점수 Δ를 실제로 내지만, 사전 등록과
출력 양쪽에서 관찰로 표시된다.

진술 셋, 아래 판정하는 자리에 인용해 두었다:

* **P1** — ``early_stopping`` / ``validation_fraction`` / ``n_estimators``·``max_iter``를 다르게
  정하는 이진 데이터셋이 몇 개인가. 2개 이상이면 이 절은 계획을 바꾼다. 0개면 프롬프트 4줄의 값을
  못 한 것이다.
* **P2** — 이 절의 예고를 실행기의 ``internal_validation``과 대조한다. 어긋나면 결과가 아니라
  **버그**이고, 실험을 중단한다.
* **P3** — 짝지은 test Δ. 관찰만.

대조군 팔은 CLI 플래그가 아니라 ``{{row_budget}}``이 빈 문자열로 렌더되는 ``main``의 git
worktree다 — 실험을 위해 더한 플래그는 그 뒤에도 제품에 남아 프롬프트를 손댈 영구적인 문이 된다.
그래서 이 스크립트는 실행 출력의 무엇으로도 두 팔을 구별할 수 없고, 하려 하지도 않는다.
``run_rowbudget.sh``가 붙인 thread-id 접두사를 믿고, 확인할 수 있는 것만 확인한다(같은 카드, 같은
분할, 같은 스레드 상태).

사용법:
    python -m bench.rowbudget                   # 모든 데이터셋
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

# 팔 둘. ``rb_off``를 앞에 두어 표가 대조군 → 처치 순으로 읽히고, P3의 Δ가 양수면 이 절이
# 도움이 된 것이다.
ARMS: tuple[tuple[str, str], ...] = (("rb_off", "rboff"), ("rb_on", "rbon"))

# P1이 세는 키. 사전 등록에서 인용한 것이고 하나도 더하지 않았다. *모델 계열*을 바꾼 계획도 달라진
# 것이지만 일부러 여기 넣지 않는다 — 기준을 판정하는 스크립트 안에서 기준을 넓히는 것이, 넓히는 일이
# 실행보다 앞서더라도, 기준이 적어 둔 뜻을 잃는 방식이다. 어차피 손해도 없다: 두 계열이 용량을 다른
# 키로 적으므로(``max_iter`` 대 ``n_estimators``) 계열 교체는 이 집합에 드러난다. 계열은 판정 옆에
# 기록해 두어 읽는 사람이 볼 수 있다.
PLAN_KEYS: tuple[str, ...] = ("early_stopping", "validation_fraction", "n_estimators", "max_iter")

# 이 절이 작용하는 iteration: 결과가 존재하기 전에 적힌 첫 계획. "우승자"가 아니다 —
# ``--max-iterations 1``에서는 같은 디렉터리지만, 어느 쪽을 뜻하는지 적어 두면 그것이 이 파일이
# 기대는 가정이 아니라 밝혀 둔 가정이 된다.
FIRST_ITERATION = 1


def run_dir(dataset: Dataset, prefix: str, seed: int = JUDGED_SEED) -> Path:
    return ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}"


def first_iteration_dir(dataset: Dataset, prefix: str, seed: int = JUDGED_SEED) -> Path:
    return run_dir(dataset, prefix, seed) / "train" / f"iter_{FIRST_ITERATION:02d}"


# --------------------------------------------------------------------------- #
# P1 — 계획
# --------------------------------------------------------------------------- #

# 여기서 ``None``은 하이퍼파라미터 값이 될 수 없으므로 "계획이 이 키를 적지 않았다"로 쓰기에
# 안전하다 — 그리고 키를 *적지 않는 것*이 바로 이 절이 만들어 내야 하는 차이다
# (침묵의 뜻은 ``early_stopping='auto'``이고, 침묵은 "off"가 아니다).
ABSENT = None


@dataclass
class Plan:
    """팔 하나의 iteration 1 계획을, P1이 견주는 것만으로 줄인 것."""

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
    """사전 등록된 키 중 두 팔이 다르게 정한 것.

    ``applied_hyperparams``가 아니라 *계획*을 본다 — 실행기가 떨어뜨린 키도 계획자가 내린 결정이고,
    P1이 묻는 것은 이 절이 계획자에게 무엇을 했는가다. 떨어뜨린 키는 함께 적어 두어, 적합에 닿지
    못한 차이가 닿은 차이로 읽히지 않게 한다.
    """
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


# --------------------------------------------------------------------------- #
# P2 — 예고를 실측과 대조
# --------------------------------------------------------------------------- #


@dataclass
class Forecast:
    """프롬프트 절이 계획자에게 말한 것. 같은 카드와 같은 함수로 다시 계산한다."""

    card_rows: int
    train: int
    auto_on: bool
    held_out_at_default: int


def forecast(card: dict[str, Any]) -> Forecast:
    """카드의 ``n_rows``에서 낸 이 절의 산술.

    보관된 프롬프트를 되파싱하지 않고, 이 절이 호출하는 것과 같은 함수인 ``row_counts``로 다시
    계산한다. 산문을 파싱하면 문구를 시험하게 된다. 이것은 수를 시험하고, P2가 말하는 것이 그것이다.
    """
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
    """실행 하나에 대한 P2 판정. ``docs/ROWBUDGET.md``에서 인용:

    > **P2 (검산).** `rb_on`의 각 실행에서, 절이 예고한 held-out 행 수와 실행기가
    > `internal_validation`에 적은 행 수가 **일치하는가.** 어긋나면 그것은 이 실험의 결과가 아니라
    > **버그**이고, 실험을 중단하고 그것을 먼저 고친다.

    확인하는 수는 하나가 아니라 둘이다. 떼어 둔 행 수는 *계획이 실제로 쓴 fraction*에서 견준다 —
    이 절의 항목은 기본값을 인용하지만 계획은 다른 값을 적어도 되고, 시험 대상은 fraction이 아니라
    카드가 함의하는 행 수다. 그리고 ``held_out + fit``을 예고된 학습 총량과 견주는데, 그쪽이 정적
    테스트가 덮을 수 없는 절반이다 — 카드의 ``n_rows``가 파일의 실제 길이와 만나는 유일한 자리다.

    결과에 ``internal_validation``이 없는 실행은 실패가 아니다. 그 적합이 아무것도 떼어 두지 않았다는
    뜻이고(early stopping이 꺼졌거나, 스스로 검증하지 않는 계열), 예고와 대조할 것이 없다. 통과가
    아니라 그대로 적는다.
    """
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


# --------------------------------------------------------------------------- #
# 데이터셋 하나
# --------------------------------------------------------------------------- #


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

    # P1을 먼저, 그리고 우승자 디렉터리가 아니라 iteration 1 디렉터리에서 읽는다.
    # ``--max-iterations 1``에서는 둘이 겹치지만, 실수로 iteration을 더 받은 실행은 그러지 않으면
    # 자기 *마지막* 계획이 다른 팔의 첫 계획과 견주어진다.
    plans = {
        arm: Plan.read(first_iteration_dir(dataset, prefix, seed), arm) for arm, prefix in ARMS
    }
    out.plan = plan_difference(plans["rb_off"], plans["rb_on"])

    # P2는 두 팔이 함께 받은 카드에 대고 본다. 카드는 하나다 — 대조군이 ``main``과 다른 것은
    # 프롬프트 템플릿이고 모델링하라고 받은 것이 아니며, 카드가 다르면 P1이 두 질문의 비교가 된다.
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

    # P3 — 관찰. 여기서부터는 ``paired.py``가 이미 쓰는 장치뿐이므로, 이 Δ와 그 문서의 Δ가
    # 어긋나면 원인은 이 파일이 아니라 분할이다.
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

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다 — ``bench/paired.py``와 같은 규칙. Δ가
    # 비워진 데이터셋의 다리를 담은 번들은, 이 판정이 일부러 내놓지 않는 수를 ``recheck``에게
    # 검산하라고 내미는 셈이 된다.
    if legs is not None:
        legs.update(scored)


# --------------------------------------------------------------------------- #
# 사전 등록된 진술
# --------------------------------------------------------------------------- #


def p1(rows: list[Row], threshold: int = 2) -> dict[str, Any]:
    """P1 판정. ``docs/ROWBUDGET.md``에서 인용:

    > **P1 (주 관찰).** 이진 4개에서, 두 팔의 iteration 1 계획이 `early_stopping` /
    > `validation_fraction` / `n_estimators`·`max_iter` 중 **하나라도 다르게 정하는 데이터셋의
    > 개수**를 센다. **2개 이상이면 이 절은 계획을 바꾼다**고 적는다. 0개면 **"프롬프트에 넣었지만
    > 계획은 같았다"**고 적는다 — 그것도 결과이고, 그러면 이 절은 프롬프트 4줄의 값을 못 한 것이다.

    셈은 하나인데 이름 붙은 결과는 둘이고 그 사이에 틈이 있다. 4개 중 1개는 "계획을 바꾼다"도
    "아무것도 안 바뀌었다"도 아니고 사전 등록이 이름을 붙이지 않았으므로, 더 좋게 읽히는 쪽으로
    둥글리지 않고 있는 수 그대로 적는다.

    이진만 본다. 못박은 4개에 못 미치는 범위는 부분 판정이라고 적는다. 퇴행은 관찰한다.
    """
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
    """P2 집계. 어디서든 어긋나면 실험을 중단한다. 결과가 아니다.

    ``rb_on``만이 아니라 두 팔을 다 확인한다. 사전 등록이 ``rb_on``을 지목한 것은 예고를 만든
    프롬프트가 그 팔의 것이기 때문이지만, 확인하는 산술은 양쪽 다 실행기의 것이고, 대조군에서
    어긋나는 것은 같은 버그를 공짜로 찾은 것이다.
    """
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
    """P3 집계. ``docs/ROWBUDGET.md``에서 인용:

    > **P3 (점수는 관찰).** test 점수의 짝지은 Δ를 데이터셋마다 내되, **판정으로 쓰지 않는다.**
    > n=5, 시드 1개, 학습 1회이고, `REPEATS.md`가 재려는 실행 간 잡음이 이 Δ에 그대로 섞여
    > 있습니다.

    ``observation_only``을 docstring에만 두지 않고 필드로 두는 이유는, JSON을 읽는 사람이 문서를
    안 읽었다는 이유로 이것을 승격시킬 수 없게 하는 것이다. 0을 벗어난 구간의 개수는 물어볼 것이므로
    적는다. 그것이 판정이 아닌 이유 옆에 적는다.
    """
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


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #


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
        # 종료 코드를 따로 두는 이유는 이것이 "실험이 아니라고 답했다"는 뜻이 아니기 때문이다.
        # 적합이 몇 행을 봤는지에 대해 예고와 실행기가 어긋났다는 뜻이고, 그것을 고치기 전까지는
        # 실행을 읽을 수 없다.
        print("P2가 어긋났으므로 P1·P3은 읽지 마세요.", file=sys.stderr)
        return 3
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
