"""커밋된 짝 판정을 예측 번들에서 다시 계산하고, 어긋나면 요란하게 알린다.

실험들이 스스로 밝힌 구멍 때문에 있다. ``docs/REPLAN.md``와 ``docs/SPG.md``는 델타와 구간을 공개하는데
그 입력인 학습된 모델은 저장소에 없고 있었던 적도 없다. 사전 등록된 기준, 바, 정지 이유, Critic 횟수는
누구나 다시 읽을 수 있었지만 수 하나도 다시 유도할 수 없었다.

그래서 판정기들은 이제 자기가 채점한 배열을 쓰고(``bench/predictions.py``) 이 모듈이 거기서 판정 전체를
다시 계산한다: leg마다의 test 점수는 같은 :func:`bench.paired.score` thunk로, 짝마다의 델타·구간·
``P(delta > 0)``은 판정 파일이 기록한 시드와 재추출 횟수로 같은 :func:`bench.paired.paired_delta`를
통해. 어디서든 어긋나면 경고가 아니라 오류다.

통과가 뜻하는 것은 정확히 이것이다: **저장된 예측과 공개된 판정 사이의 산술이 맞다.** 그 예측이 config가
가리키는 모델에서 나왔다는 것은 보여주지 않는다 — 그건 여전히 실행 보관본에 달려 있다. 좁은 주장을
이름 붙이는 것이 요점이다. 검사한 것보다 많은 것을 암시하는 검사기는 없는 것보다 나쁘다.

판정기마다 판정 파일 옆에 번들을 쓰므로 어느 것이든 읽는다 — 그리고 인자가 없으면 전부 읽는다. 벤치마크
문서들이 하는 주장이 그것이기 때문이다:

    python -m bench.recheck                                    # bench/runs/paired/의 판정 전부
    python -m bench.recheck bench/runs/paired/seed42.json      # RESULTS.md
    python -m bench.recheck bench/runs/paired/spg-seed42.json  # SPG.md

인자 없는 형태가 파일 하나가 아니라 디렉터리 전체를 기본값으로 삼는 것은 의도다. 파일 하나가 기본이면
"``python -m bench.recheck``가 통과한다"가 그 파일에 대한 문장이 되는데, 문서들은 파일 하나를 주장하지
않는다 — 자기가 공개하는 델타 전부를 주장한다. 그래서 기본값은 집합이고, 새 판정 파일은 존재하기만 하면
거기 들어오고, 번들 없는 판정은 집합 밖에 앉는 대신 기본 호출을 실패시킨다. 판정기가 번들을 낼 줄 알기
전에 쓰인 판정에는 번들이 없고, 그건 통과가 아니라 exit 2다 — 그 판정기를 다시 돌리면 생긴다.

**여기서 공허하게 통과할 수 있는 길은 없다.** 대조할 수가 하나도 없이 끝나는 경로는 전부 exit 0이 아니라
exit 2다: 최상위 모양이 :data:`ENTRY_KEYS`에 없는 파일, 판정이 거부하지 않았는데 번들에 leg가 없는
데이터셋, 이 모듈이 읽지 않는 키 아래에 델타가 있는 항목, 데이터셋이 전부 건너뛰어진 실행. 가정이 아니다 —
이전 판이 그렇게 했다. ``datasets[].pairs``만 읽었으므로 ``rowbudget-seed42.json``(델타 다섯 개가
``datasets[].delta`` 아래)은 건너뛴 다섯 행으로 처리되고 ``repeats-seed42.json``(열다섯 개가
``repeat_datasets[].pairs`` 아래)은 0으로 합산됐고, 둘 다 일치한다고 출력했다. 읽지도 않은 수에 대해
일치를 보고하는 검사기는 검사기가 없는 것보다 나쁘다 — 없던 시절은 적어도 아니라고 주장하지는 않았다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bench.paired import OUT_DIR, SCORE_TOLERANCE, Scored, paired_delta, score
from bench.predictions import bundle_path, read_bundle

# 같은 코드, 같은 배열, 같은 시드면 마지막 비트까지 맞아야 한다. 문턱은 BLAS나 numpy 빌드가 달라도
# 살아남을 만큼 느슨하게 두고, 실제로 본 가장 큰 차이는 어느 쪽이든 출력한다 — "허용 오차 안"만
# 말하는 검사기는 1e-16과 9e-7의 차이를 숨기고, 그 둘은 같은 소식이 아니다.
DELTA_TOLERANCE = 1e-9

FIELDS = ("delta", "ci_low", "ci_high", "p_better")

# 판정기들이 델타를 한 키 아래에만 공개하지는 않는다. 가장 흔한 모양만 아는 검사기는 못 읽는 파일에
# 대해 일치를 보고하게 된다. 그래서 모양을 나열하고, 목록 밖은 건너뛰지 않고 거부한다:
#
#   datasets[].pairs[]         seed4N, replan, spg  — 데이터셋마다 델타 목록
#   datasets[].delta           rowbudget            — 데이터셋마다 델타 정확히 하나
#   repeat_datasets[].pairs[]  repeats              — 자기 키 아래의 두 번째 데이터셋 목록
#
# 판정기를 더하면 그 키를 여기 더해야 하고, 잊으면 exit 2다.
ENTRY_KEYS = ("datasets", "repeat_datasets")


class Mismatch(Exception):
    """다시 계산한 수가 커밋된 것과 맞지 않는다."""


class UnknownVerdictShape(Exception):
    """이 모듈이 읽을 수 없는, 그래서 통과시키지 않을 판정 파일."""


def published_deltas(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """데이터셋 항목 하나가 공개하는 델타 전부. 어느 키 아래에 있든."""
    out = [pair for pair in entry.get("pairs") or [] if isinstance(pair, dict)]
    single = entry.get("delta")
    if isinstance(single, dict):
        out.append(single)
    return out


def dataset_entries(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """판정 파일의 데이터셋 항목 전부.

    :data:`ENTRY_KEYS` 중 아무것도 없으면 :class:`UnknownVerdictShape`를 낸다. 빈 목록과 이 모듈이
    모르는 모양은 루프 안에서 똑같이 보이고, 둘 중 하나만 통과할 수 있다.
    """
    present = [key for key in ENTRY_KEYS if isinstance(verdict.get(key), list)]
    if not present:
        raise UnknownVerdictShape(
            f"이 판정 파일에는 {' · '.join(ENTRY_KEYS)} 중 어느 것도 없습니다 "
            f"(최상위 키: {', '.join(sorted(verdict))}). 이 모듈이 모르는 형태이므로 "
            "0개를 대조하고 통과시키는 대신 여기서 멈춥니다 — 새 판정 형태라면 "
            "bench/recheck.py의 ENTRY_KEYS에 키를 추가하세요"
        )
    return [entry for key in present for entry in verdict[key] if isinstance(entry, dict)]


def _agrees(recomputed: float, recorded: Any, tolerance: float) -> tuple[bool, float]:
    """두 수가 맞는지. nan은 자기 자신에게만 맞는 값으로 다룬다.

    ``identical_predictions``인 짝은 설계상 nan 경계를 가지므로 nan 대 nan은 통과, nan 대 수는
    보이는 그대로 실패다.
    """
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
        return False, float("nan")
    a, b = float(recomputed), float(recorded)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b), float("nan")
    diff = abs(a - b)
    return diff <= tolerance, diff


def check_dataset(
    entry: dict[str, Any], legs: dict[str, Scored], seed: int, resamples: int
) -> dict[str, Any]:
    """데이터셋 하나의 leg와 짝을 다시 계산한다. 어긋나면 :class:`Mismatch`."""
    dataset = entry["dataset"]
    metric = entry["metric"]
    worst = 0.0
    checked_legs = 0
    checked_pairs = 0

    rows = {arm: np.asarray(leg.y_test) for arm, leg in legs.items()}
    reference_arm, reference = next(iter(rows.items()))
    for arm, y_test in rows.items():
        if not np.array_equal(y_test, reference):
            raise Mismatch(
                f"{dataset}: {arm}의 정답 배열이 {reference_arm}과 다릅니다 — 같은 test 행이 아닙니다"
            )

    recorded_fingerprint = entry.get("test_fingerprint")
    for arm, leg in legs.items():
        if recorded_fingerprint and leg.fingerprint != recorded_fingerprint:
            raise Mismatch(
                f"{dataset}: {arm}의 지문 {leg.fingerprint[:12]}이 판정 파일의 "
                f"{str(recorded_fingerprint)[:12]}과 다릅니다"
            )
        recomputed = score(metric, leg.y_test, leg.pred, leg.proba, leg.average, leg.task)
        recorded = entry.get("arms", {}).get(arm, {}).get("test_reproduced")
        ok, diff = _agrees(recomputed, recorded, SCORE_TOLERANCE)
        if not ok:
            raise Mismatch(
                f"{dataset}/{arm}: 다시 계산한 {metric} {recomputed:.8f}이 판정 파일의 "
                f"{recorded}와 다릅니다"
            )
        worst = max(worst, 0.0 if math.isnan(diff) else diff)
        checked_legs += 1

    for pair in published_deltas(entry):
        a_name, b_name = pair["a"], pair["b"]
        if a_name not in legs or b_name not in legs:
            raise Mismatch(f"{dataset}: {a_name} 또는 {b_name}의 예측이 번들에 없습니다")
        again = paired_delta(legs[a_name], legs[b_name], metric, resamples, seed, pair["kind"])
        if bool(again.identical_predictions) != bool(pair.get("identical_predictions")):
            raise Mismatch(
                f"{dataset}/{pair['kind']}: 예측 동일 여부가 판정 파일과 다릅니다 "
                f"({again.identical_predictions} 대 {pair.get('identical_predictions')})"
            )
        if int(again.resamples_used) != int(pair.get("resamples_used", -1)):
            raise Mismatch(
                f"{dataset}/{pair['kind']}: 쓸 수 있었던 재추출이 {again.resamples_used}회인데 "
                f"판정 파일은 {pair.get('resamples_used')}회입니다"
            )
        for field in FIELDS:
            ok, diff = _agrees(getattr(again, field), pair.get(field), DELTA_TOLERANCE)
            if not ok:
                raise Mismatch(
                    f"{dataset}/{pair['kind']}: {field}가 {getattr(again, field)}인데 판정 "
                    f"파일은 {pair.get(field)}입니다"
                )
            worst = max(worst, 0.0 if math.isnan(diff) else diff)
        checked_pairs += 1

    return {"dataset": dataset, "legs": checked_legs, "pairs": checked_pairs, "max_diff": worst}


def recheck(verdict_path: Path, bundle: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """판정 파일의 모든 데이터셋을 다시 계산한다. ``(rows, problems)``를 돌려준다.

    이 모듈이 읽을 수 없는 파일에는 :class:`UnknownVerdictShape`를 올린다. 건너뛴 행이 될 수 있는
    것은 판정 자신이 ``refused``라고 적은 데이터셋뿐이고, 비교를 내지 못하는 그 밖의 모든 것은
    문제다.
    """
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    legs_by_dataset, _meta = read_bundle(bundle or bundle_path(verdict_path))
    seed = int(verdict["seed"])
    resamples = int(verdict["resamples"])

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for entry in dataset_entries(verdict):
        dataset = entry["dataset"]
        published = published_deltas(entry)
        legs = legs_by_dataset.get(dataset)
        # 거부는 빈 ``pairs`` 목록이 아니라 ``verdict``에서 읽는다. 판정기가 델타를 다른 키 아래
        # 공개한 순간 그 둘이 갈라졌다: ``rowbudget``의 ``verdict: ok`` 항목 다섯은 ``pairs``가
        # 없어서 "건너뜀 (판정=ok)"으로 처리됐다 — 이 코드가 예상한 적 없는 건너뛰기 이유가
        # 일치인 것처럼 출력됐다.
        if entry.get("verdict") == "refused":
            if published:
                problems.append(
                    f"{dataset}: 판정이 refused인데 Δ를 {len(published)}개 공개했습니다 — "
                    "거부는 아무것도 공개하지 않아야 합니다"
                )
            else:
                rows.append(
                    {
                        "dataset": dataset,
                        "legs": 0,
                        "pairs": 0,
                        "max_diff": 0.0,
                        "skipped": entry.get("verdict"),
                    }
                )
            continue
        if not legs:
            # 판정이 이 데이터셋을 거부하지 않았으므로 자기가 공개한 것을 책임지고, 번들이 그걸
            # 설명해야 한다. 델타가 0개라고 무해해지지 않는다 — 수가 이 모듈이 읽지 않는 어딘가에
            # 있다는 뜻이다.
            problems.append(
                f"{dataset}: 판정이 {entry.get('verdict')!r}인데 번들에 예측이 없습니다 "
                f"(공개된 Δ {len(published)}개)"
            )
            continue
        try:
            row = check_dataset(entry, legs, seed, resamples)
        except Mismatch as exc:
            problems.append(str(exc))
            continue
        if not row["pairs"]:
            problems.append(
                f"{dataset}: 다리 {row['legs']}개의 점수는 맞지만 대조한 Δ가 0개입니다 — "
                f"거부가 아닌 판정({entry.get('verdict')!r})이 공개한 Δ가 이 모듈이 읽는 키 "
                f"({' · '.join(('pairs', 'delta'))}) 밖에 있습니다"
            )
        rows.append(row)
    return rows, problems


def all_verdicts() -> list[Path]:
    """:data:`bench.paired.OUT_DIR`의 판정 파일 전부, 정렬해서.

    그 디렉터리의 모든 ``*.json``이 판정이다 — 번들은 옆에 ``.npz``로 앉는다. 그래서 새 판정기의 출력은
    존재하기만 하면 기본 호출에 들어오고, 그게 요점이다: 손으로 유지하는 목록은 누군가 추가를 잊는
    목록이고, 그 누락은 통과처럼 보인다.
    """
    return sorted(OUT_DIR.glob("*.json"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="커밋된 짝지은 판정을 예측 번들에서 다시 계산해 대조합니다 (모델·원본 데이터 불필요)"
    )
    parser.add_argument(
        "verdict",
        nargs="?",
        type=Path,
        default=None,
        help=f"판정 파일 (기본: {OUT_DIR.as_posix()}/의 판정 파일 전부)",
    )
    parser.add_argument("--bundle", type=Path, default=None, help="기본: 판정 파일 옆의 .predictions.npz")
    return parser.parse_args(argv)


def check_one(verdict: Path, bundle: Path | None = None) -> tuple[int, int, int]:
    """판정 파일 하나를 다시 대조하고 그 행들을 출력한다.

    ``(exit_code, legs_checked, pairs_checked)``를 돌려준다. 코드가 0이 아니면 두 개수는 0이므로,
    디렉터리 전체에서 합산하는 호출자가 실패한 파일을 실수로 계산에 넣을 수 없다.
    """
    if not verdict.exists():
        print(f"판정 파일이 없습니다: {verdict}", file=sys.stderr)
        return 2, 0, 0
    target = bundle or bundle_path(verdict)
    if not target.exists():
        print(
            f"예측 번들이 없습니다: {target}\n"
            "판정을 다시 내면 함께 만들어집니다 (python -m bench.replan)",
            file=sys.stderr,
        )
        return 2, 0, 0

    print(f"재계산 대조 · 판정 {verdict.as_posix()} · 번들 {target.as_posix()}")
    try:
        rows, problems = recheck(verdict, target)
    except UnknownVerdictShape as exc:
        print(str(exc), file=sys.stderr)
        return 2, 0, 0
    for row in rows:
        if row.get("skipped"):
            print(f"   {row['dataset']:<14} 건너뜀 (판정={row['skipped']})")
        else:
            print(
                f"   {row['dataset']:<14} 다리 {row['legs']}개 · 쌍 {row['pairs']}개 일치 · "
                f"최대 차이 {row['max_diff']:.3e}"
            )
    if problems:
        print()
        for problem in problems:
            print(f"불일치: {problem}", file=sys.stderr)
        return 1, 0, 0
    total_pairs = sum(int(r["pairs"]) for r in rows)
    total_legs = sum(int(r["legs"]) for r in rows)
    if not total_pairs:
        # 어긋난 것이 없음과 대조한 것이 없음은 같은 결과가 아니고, 두 번째는 통과가 아니다.
        # 공허한 경로 중 마지막이고, 파일의 데이터셋이 모두 정당하게 거부됐을 때 닿는 유일한
        # 경로다.
        print(
            f"대조한 Δ가 0개입니다 (데이터셋 {len(rows)}개가 모두 건너뛰어졌습니다). "
            "통과가 아니라 확인 불가입니다 — 이 판정 파일은 검산할 수를 공개하지 않았습니다.",
            file=sys.stderr,
        )
        return 2, 0, 0
    print()
    print(f"다리 {total_legs}개의 점수와 쌍 {total_pairs}개의 Δ·CI·P(Δ>0)가 판정 파일과 일치합니다.")
    return 0, total_legs, total_pairs


NARROW_CLAIM = (
    "이것이 확인하는 것은 저장된 예측과 공개된 수 사이의 산술이고, 그 예측이 config가 "
    "가리키는 모델에서 나왔다는 것은 아닙니다."
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verdict is not None:
        code, _legs, _pairs = check_one(args.verdict, args.bundle)
        if code == 0:
            print(NARROW_CLAIM)
        return code

    if args.bundle is not None:
        print("--bundle 은 판정 파일 하나를 함께 줄 때만 씁니다", file=sys.stderr)
        return 2
    verdicts = all_verdicts()
    if not verdicts:
        # 빈 디렉터리는 아무것도 대조하지 않고, 대조한 것이 없으면 통과가 아니다. 데이터셋이
        # 전부 건너뛰어진 파일과 같은 규칙이다.
        print(
            f"{OUT_DIR.as_posix()}/에 판정 파일이 없습니다. 통과가 아니라 확인 불가입니다.",
            file=sys.stderr,
        )
        return 2

    results = []
    for verdict in verdicts:
        results.append(check_one(verdict, None))
        print()
    failed = [v.name for v, (code, _, _) in zip(verdicts, results, strict=True) if code != 0]
    if failed:
        print(
            f"판정 파일 {len(verdicts)}개 중 {len(failed)}개가 대조되지 않았습니다: "
            f"{' · '.join(failed)}",
            file=sys.stderr,
        )
        return max(code for code, _, _ in results)
    print(
        f"판정 파일 {len(verdicts)}개 · 다리 {sum(legs for _, legs, _ in results)}개 · "
        f"쌍 {sum(pairs for _, _, pairs in results)}개가 전부 일치합니다. {NARROW_CLAIM}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
