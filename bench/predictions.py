"""Save the arrays behind each paired verdict, so it can be rechecked.

Roles:

* Bundle layout — key names and the bundle file path.
* Writing — save every leg's arrays into one npz.
* Reading — load a bundle back into ``Scored`` legs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.paired import Scored, Winner

# --- Role: bundle layout -------------------------------------------------------------

# Not "/": some zip tools read it as a folder.
SEP = "::"
META_KEY = "__meta__"
BUNDLE_SUFFIX = ".predictions.npz"


def bundle_path(verdict_path: Path) -> Path:
    """Bundle file next to a verdict file, derived so they always match."""
    return verdict_path.with_suffix("").with_suffix(BUNDLE_SUFFIX)


# --- Role: writing ------------------------------------------------------------------


def _leg_meta(dataset: str, arm: str, metric: str, scored: Scored) -> dict[str, Any]:
    """_leg_meta | Writing: the manifest entry for one leg."""
    return {
        "dataset": dataset,
        "arm": arm,
        "metric": metric,
        "task": scored.task,
        "average": scored.average,
        "n_classes": scored.n_classes,
        "n_rows": scored.n_rows,
        "label": scored.winner.label,
        "fits": scored.winner.fits,
        "test_score": scored.test_score,
        "fingerprint": scored.fingerprint,
        "has_proba": scored.proba is not None,
    }


def write_bundle(
    path: Path, legs: dict[str, dict[str, Scored]], metrics: dict[str, str]
) -> dict[str, Any]:
    """Write every leg into one compressed npz; return its manifest.

    Arm keys must match the verdict's ``a``/``b``. Arrays keep full width."""
    arrays: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for dataset in sorted(legs):
        metric = metrics[dataset]
        for arm in sorted(legs[dataset]):
            scored = legs[dataset][arm]
            stem = f"{dataset}{SEP}{arm}{SEP}"
            arrays[stem + "pred"] = np.asarray(scored.pred)
            arrays[stem + "y"] = np.asarray(scored.y_test)
            if scored.proba is not None:
                arrays[stem + "proba"] = np.asarray(scored.proba, dtype=np.float64)
            manifest.append(_leg_meta(dataset, arm, metric, scored))
    meta = {"format": 1, "generated_by": "bench/predictions.py", "legs": manifest}
    arrays[META_KEY] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return meta


# --- Role: reading ------------------------------------------------------------------


def read_bundle(path: Path) -> tuple[dict[str, dict[str, Scored]], dict[str, Any]]:
    """Rebuild the ``Scored`` legs; return ``(legs, meta)``.

    ``Winner`` is a shell: no paths or config are stored."""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data:
            raise ValueError(f"{path}: {META_KEY}가 없습니다 — bench/predictions.py가 쓴 파일이 아닙니다")
        meta = json.loads(str(data[META_KEY].item()))
        legs: dict[str, dict[str, Scored]] = {}
        for entry in meta["legs"]:
            dataset, arm = entry["dataset"], entry["arm"]
            stem = f"{dataset}{SEP}{arm}{SEP}"
            proba = data[stem + "proba"] if entry["has_proba"] else None
            legs.setdefault(dataset, {})[arm] = Scored(
                winner=Winner(
                    arm=arm,
                    label=entry["label"],
                    directory=Path("(번들에는 경로가 없습니다)"),
                    config={},
                    val_score=float("nan"),
                    recorded_test=float("nan"),
                    fits=int(entry["fits"]),
                ),
                pred=data[stem + "pred"],
                y_test=data[stem + "y"],
                test_score=float(entry["test_score"]),
                fingerprint=str(entry["fingerprint"]),
                task=str(entry["task"]),
                n_classes=int(entry["n_classes"]),
                proba=proba,
                average=str(entry["average"]),
            )
    return legs, meta
