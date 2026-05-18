#!/usr/bin/env bash
# Quick sanity check: 128 envs, 20 iterations.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
recovery_require_motion

RUN_NAME="${MJLAB_RUN_NAME:-recovery_smoke}"
NUM_ENVS="${MJLAB_NUM_ENVS:-128}"
MAX_ITERS="${MJLAB_MAX_ITERS:-20}"

echo "[recovery] smoke train: ${RUN_NAME} (${NUM_ENVS} envs, ${MAX_ITERS} iters)"
recovery_uv train "${TASK_ID}" \
  --env.teacher.motion-path "${TEACHER_MOTION_PATH}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.max-iterations "${MAX_ITERS}" \
  --agent.run-name "${RUN_NAME}"
