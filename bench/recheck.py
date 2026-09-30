"""Recompute committed paired verdicts from prediction bundles; fail loudly on mismatch.

Roles:

* Verdict shapes — tolerances, known entry keys, and error types.
* Recompute — redo leg scores and pair deltas per dataset.
* CLI — check one file or every verdict file.

Usage:
    python -m bench.recheck
    python -m bench.recheck bench/runs/paired/seed42.json
    python -m bench.recheck bench/runs/paired/spg-seed42.json
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

# --- Role: verdict shapes -------------------------------------------------------------

# Loose enough for other BLAS builds; worst diff is printed.
DELTA_TOLERANCE = 1e-9

FIELDS = ("delta", "ci_low", "ci_high", "p_better")

# Known top-level keys; any other shape exits 2, never passes.
ENTRY_KEYS = ("datasets", "repeat_datasets")


class Mismatch(Exception):
    """A recomputed number does not match the committed one."""


class UnknownVerdictShape(Exception):
    """A verdict file this module cannot read, so it never passes."""


def published_deltas(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """All deltas one dataset entry publishes, under ``pairs`` or ``delta``."""
    out = [pair for pair in entry.get("pairs") or [] if isinstance(pair, dict)]
    single = entry.get("delta")
    if isinstance(single, dict):
        out.append(single)
    return out


def dataset_entries(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """All dataset entries of a verdict file; unknown shapes raise UnknownVerdictShape."""
    present = [key for key in ENTRY_KEYS if isinstance(verdict.get(key), list)]
    if not present:
        raise UnknownVerdictShape(
            f"이 판정 파일에는 {' · '.join(ENTRY_KEYS)} 중 어느 것도 없습니다 "
            f"(최상위 키: {', '.join(sorted(verdict))}). 이 모듈이 모르는 형태이므로 "
            "0개를 대조하고 통과시키는 대신 여기서 멈춥니다 — 새 판정 형태라면 "
            "bench/recheck.py의 ENTRY_KEYS에 키를 추가하세요"
        )
    return [entry for key in present for entry in verdict[key] if isinstance(entry, dict)]


# --- Role: recompute ------------------------------------------------------------------


def _agrees(recomputed: float, recorded: Any, tolerance: float) -> tuple[bool, float]:
    """_agrees | Recompute: match within tolerance; nan matches only nan."""
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
    """Recompute one dataset's legs and pairs; raise Mismatch on any gap."""
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
    """Recompute every dataset of a verdict file; return ``(rows, problems)``.

    Only datasets marked ``refused`` may be skipped; all else is a problem."""
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
        # Read refusal from ``verdict``, not from empty ``pairs``.
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
            # Not refused, so the bundle must hold its legs.
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


# --- Role: CLI ------------------------------------------------------------------------


def all_verdicts() -> list[Path]:
    """Every verdict ``*.json`` in OUT_DIR, sorted; no hand-kept list."""
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
    """Recheck one verdict file and print its rows.

    Return ``(exit_code, legs, pairs)``; counts are 0 on failure."""
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
        # Nothing compared is not a pass: exit 2.
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
        # Empty directory compares nothing, so it is not a pass.
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
