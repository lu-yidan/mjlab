#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Recovery log root: ${LOG_ROOT}"
echo ""
printf "%-12s  %-55s  %s\n" "LATEST_CKPT" "RUN_DIR" "CHECKPOINTS"
printf "%s\n" "------------------------------------------------------------------------"

for run_dir in "${LOG_ROOT}"/*; do
  [[ -d "${run_dir}" ]] || continue
  latest_ckpt="$(ls -1 "${run_dir}"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "${latest_ckpt}" ]]; then
    continue
  fi
  ckpt_count="$(ls -1 "${run_dir}"/model_*.pt 2>/dev/null | wc -l)"
  printf "%-12s  %-55s  %s\n" \
    "$(basename "${latest_ckpt}")" \
    "$(basename "${run_dir}")" \
    "${ckpt_count} files"
done
