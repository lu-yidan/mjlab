#!/usr/bin/env bash
# Extract still frames from a play video for quick qualitative review.
#
# Usage:
#   ./scripts/recovery/extract_frames.sh [video.mp4]
#   ./scripts/recovery/extract_frames.sh   # uses latest mp4 under default run
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_DIR="$(recovery_resolve_run_dir "${MJLAB_RUN_DIR:-}")"
VIDEO="${1:-}"
if [[ -z "${VIDEO}" ]]; then
  VIDEO="$(ls -1t "${RUN_DIR}/videos/play/"*.mp4 2>/dev/null | head -1 || true)"
fi
if [[ -z "${VIDEO}" || ! -f "${VIDEO}" ]]; then
  echo "[recovery] no video found. Run play_video.sh first." >&2
  exit 1
fi

OUT_DIR="${RUN_DIR}/videos/play/frames"
mkdir -p "${OUT_DIR}"

TIMES="${MJLAB_FRAME_TIMES:-1.0 2.5 4.0}"
idx=0
for t in ${TIMES}; do
  idx=$((idx + 1))
  out="${OUT_DIR}/frame_${idx}_${t}s.png"
  ffmpeg -y -loglevel error -ss "${t}" -i "${VIDEO}" -frames:v 1 "${out}"
  echo "[recovery] ${out}"
done

echo "[recovery] frames in ${OUT_DIR}"
