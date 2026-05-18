#!/usr/bin/env bash
# Full posture-polish training from scratch (4096 envs, 6000 iters).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
recovery_require_motion

RUN_NAME="${MJLAB_RUN_NAME:-late_phase_polish_full}"
NUM_ENVS="${MJLAB_NUM_ENVS:-4096}"
MAX_ITERS="${MJLAB_MAX_ITERS:-6000}"

echo "[recovery] polish train: ${RUN_NAME}"
echo "  envs=${NUM_ENVS}  iters=${MAX_ITERS}"
echo "  motion=${TEACHER_MOTION_PATH}"

recovery_uv train "${TASK_ID}" \
  --env.teacher.motion-path "${TEACHER_MOTION_PATH}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${MAX_ITERS}" \
  --agent.run-name "${RUN_NAME}"
