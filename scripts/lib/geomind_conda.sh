#!/usr/bin/env bash

# Override CONDA_SH for a custom installation; CONDA_ENV_NAME defaults to isaac_curobo.
geomind_activate_conda() {
  local env_name="${1:-${CONDA_ENV_NAME:-isaac_curobo}}"
  local conda_sh="${CONDA_SH:-}" conda_base="" candidate
  if [[ -n "${conda_sh}" ]]; then
    if [[ ! -f "${conda_sh}" ]]; then
      echo "[geomind] ERROR: CONDA_SH does not exist: ${conda_sh}" >&2
      return 1
    fi
  else
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      conda_base="$("${CONDA_EXE}" info --base 2>/dev/null)" || conda_base=""
    elif command -v conda >/dev/null 2>&1; then
      conda_base="$(conda info --base 2>/dev/null)" || conda_base=""
    fi
    for candidate in \
      "${conda_base:+${conda_base}/etc/profile.d/conda.sh}" \
      "${HOME:-}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME:-}/anaconda3/etc/profile.d/conda.sh" \
      "${HOME:-}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME:-}/mambaforge/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -n "${candidate}" && -f "${candidate}" ]]; then
        conda_sh="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${conda_sh}" ]]; then
    echo "[geomind] ERROR: Conda initialization not found. Set CONDA_SH=/path/to/etc/profile.d/conda.sh." >&2
    return 1
  fi
  source "${conda_sh}" || {
    echo "[geomind] ERROR: Cannot initialize Conda from ${conda_sh}." >&2
    return 1
  }
  conda activate "${env_name}" || {
    echo "[geomind] ERROR: Cannot activate '${env_name}'. Create it or set CONDA_ENV_NAME (or the script's environment override)." >&2
    return 1
  }
}
