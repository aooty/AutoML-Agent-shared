#!/usr/bin/env bash
# The 36 fits behind docs/WEIGHT-LEVER.md; CPU only, no API.
# Three treatments per cell: ratio, balanced, and refit.
# Threads pinned: the refit control must not move.
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$(dirname "$0")/../.."

echo "=== 서른여섯 번 학습 ($(date '+%H:%M:%S')) ==="
python -m bench.weight_lever --fit
fit_status=$?
echo "--- 학습 exit=${fit_status} ($(date '+%H:%M:%S')) ---"

if [ "$fit_status" -ne 0 ]; then
  echo "=== 학습에 실패한 칸이 있습니다 — 판정하지 않습니다 ==="
  echo "=== bench/runs/weightlever/fits.json을 보십시오 ==="
  exit "$fit_status"
fi

echo "=== 판정 ($(date '+%H:%M:%S')) ==="
python -m bench.weight_lever
echo "=== 끝났습니다 ($(date '+%H:%M:%S')) ==="
