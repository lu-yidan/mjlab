#!/usr/bin/env bash
# Kill mjlab train/play processes on a GPU (default: CUDA_VISIBLE_DEVICES).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GPU="${CUDA_DEVICE}"
echo "[recovery] killing mjlab uv processes on GPU ${GPU}..."

pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || true)"
if [[ -z "${pids}" ]]; then
  echo "[recovery] no GPU processes on device ${GPU}"
  exit 0
fi

for pid in ${pids}; do
  cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  if [[ "${cmd}" == *"uv run"* && "${cmd}" == *"mjlab"* ]]; then
    echo "  kill ${pid}: ${cmd}"
    kill "${pid}" || true
  fi
done

echo "[recovery] done"
