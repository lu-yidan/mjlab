#!/usr/bin/env bash
# Fine-tune robustness from a polished checkpoint (default: polish run @ 4400).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
recovery_require_motion

LOAD_RUN="${MJLAB_LOAD_RUN:-${DEFAULT_POLISH_RUN}}"
LOAD_CKPT="${MJLAB_LOAD_CHECKPOINT:-model_4400.pt}"
RUN_NAME="${MJLAB_RUN_NAME:-late_phase_push_robust}"
NUM_ENVS="${MJLAB_NUM_ENVS:-4096}"
MAX_ITERS="${MJLAB_MAX_ITERS:-6000}"

echo "[recovery] robust resume train: ${RUN_NAME}"
echo "  resume=${LOAD_RUN}/${LOAD_CKPT}"
echo "  envs=${NUM_ENVS}  iters=${MAX_ITERS}"

recovery_uv train "${TASK_ID}" \
  --env.teacher.motion-path "${TEACHER_MOTION_PATH}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --agent.resume True \
  --agent.load-run "${LOAD_RUN}" \
  --agent.load-checkpoint "${LOAD_CKPT}" \
  --agent.max-iterations "${MAX_ITERS}" \
  --agent.run-name "${RUN_NAME}"
