#!/usr/bin/env bash
# Shared defaults for G1 recovery train/play helpers.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

_recovery_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MJLAB_REPO_ROOT="$(cd "${_recovery_common_dir}/../.." && pwd)"

TASK_ID="Mjlab-Recovery-Flat-Unitree-G1"
TEACHER_MOTION_PATH="${MJLAB_TEACHER_MOTION:-${MJLAB_REPO_ROOT}/artifacts/recovery_motions/g1_amp_get_up}"
LOG_ROOT="${MJLAB_LOG_ROOT:-${MJLAB_REPO_ROOT}/logs/rsl_rl/g1_recovery}"
NUM_ENVS="${MJLAB_NUM_ENVS:-4096}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# Recent reference runs (override with MJLAB_RUN_DIR / MJLAB_CHECKPOINT).
DEFAULT_POLISH_RUN="2026-04-15_11-07-35_late_phase_polish_full_v1"
DEFAULT_ROBUST_RUN="2026-04-15_16-03-46_late_phase_push_robust_v2_resume4400"
DEFAULT_CHECKPOINT="${MJLAB_CHECKPOINT:-model_5999.pt}"

recovery_cd_repo() {
  cd "${MJLAB_REPO_ROOT}" || exit 1
}

recovery_require_motion() {
  if [[ ! -d "${TEACHER_MOTION_PATH}" ]]; then
    echo "[recovery] teacher motion dir not found: ${TEACHER_MOTION_PATH}" >&2
    echo "Set MJLAB_TEACHER_MOTION or convert clips with amp_pkl_to_npz.py" >&2
    exit 1
  fi
}

recovery_resolve_run_dir() {
  local run_name="${1:-${MJLAB_RUN_DIR:-${DEFAULT_ROBUST_RUN}}}"
  if [[ -d "${run_name}" ]]; then
    echo "${run_name}"
    return
  fi
  if [[ -d "${LOG_ROOT}/${run_name}" ]]; then
    echo "${LOG_ROOT}/${run_name}"
    return
  fi
  echo "[recovery] run directory not found: ${run_name}" >&2
  exit 1
}

recovery_resolve_checkpoint() {
  local run_dir
  run_dir="$(recovery_resolve_run_dir "$1")"
  local ckpt="${2:-${MJLAB_CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
  if [[ -f "${ckpt}" ]]; then
    echo "${ckpt}"
    return
  fi
  if [[ -f "${run_dir}/${ckpt}" ]]; then
    echo "${run_dir}/${ckpt}"
    return
  fi
  echo "[recovery] checkpoint not found: ${ckpt} (run: ${run_dir})" >&2
  exit 1
}

recovery_uv() {
  recovery_cd_repo
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" uv run "$@"
}
