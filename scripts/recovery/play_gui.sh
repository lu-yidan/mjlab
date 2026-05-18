#!/usr/bin/env bash
# Interactive play with native or viser viewer.
#
# Examples:
#   ./scripts/recovery/play_gui.sh
#   MJLAB_RUN_DIR=2026-04-15_16-03-46_late_phase_push_robust_v2_resume4400 \
#     MJLAB_CHECKPOINT=model_5000.pt ./scripts/recovery/play_gui.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CKPT_PATH="$(recovery_resolve_checkpoint "${MJLAB_RUN_DIR:-}" "${MJLAB_CHECKPOINT:-}")"
VIEWER="${MJLAB_VIEWER:-auto}"
NUM_ENVS="${MJLAB_NUM_ENVS:-1}"
USE_TEACHER_RESET="${MJLAB_TEACHER_RESET:-1}"

echo "[recovery] play ${CKPT_PATH}"
echo "  viewer=${VIEWER}  num_envs=${NUM_ENVS}"

args=(
  play "${TASK_ID}"
  --checkpoint-file "${CKPT_PATH}"
  --num-envs "${NUM_ENVS}"
  --viewer "${VIEWER}"
  --no-terminations True
)

if [[ "${USE_TEACHER_RESET}" == "1" ]]; then
  recovery_require_motion
  args+=(--teacher-motion-path "${TEACHER_MOTION_PATH}")
fi

recovery_uv "${args[@]}"
