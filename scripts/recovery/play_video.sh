#!/usr/bin/env bash
# Headless play + mp4 recording (viser viewer, exits after video_length steps).
#
# Examples:
#   ./scripts/recovery/play_video.sh
#   MJLAB_CHECKPOINT=model_4500.pt MJLAB_VIDEO_LENGTH=300 ./scripts/recovery/play_video.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
recovery_require_motion

CKPT_PATH="$(recovery_resolve_checkpoint "${MJLAB_RUN_DIR:-}" "${MJLAB_CHECKPOINT:-}")"
RUN_DIR="$(dirname "${CKPT_PATH}")"
VIDEO_LENGTH="${MJLAB_VIDEO_LENGTH:-240}"
VIEWER="${MJLAB_VIEWER:-viser}"

echo "[recovery] record video from ${CKPT_PATH}"
echo "  length=${VIDEO_LENGTH} frames  viewer=${VIEWER}"
echo "  output -> ${RUN_DIR}/videos/play/"

recovery_uv play "${TASK_ID}" \
  --checkpoint-file "${CKPT_PATH}" \
  --num-envs 1 \
  --viewer "${VIEWER}" \
  --video True \
  --video-length "${VIDEO_LENGTH}" \
  --no-terminations True \
  --teacher-motion-path "${TEACHER_MOTION_PATH}"

echo ""
echo "[recovery] done. video(s):"
ls -1 "${RUN_DIR}/videos/play/"*.mp4 2>/dev/null || echo "  (no mp4 found yet)"
