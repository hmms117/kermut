#!/usr/bin/env bash
set -euo pipefail

# Resolve the project directory and requirements file.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements_cpu.txt"
PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Could not find requirements file at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

# Allow overriding defaults via environment variables.
UV_BIN="${UV_BIN:-uv}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

# Ensure ~/.local/bin is considered when searching for uv.
export PATH="${HOME}/.local/bin:${PATH}"

ensure_uv() {
  if command -v "${UV_BIN}" >/dev/null 2>&1; then
    UV_BIN="$(command -v "${UV_BIN}")"
    return
  fi

  echo "uv was not found on PATH. Installing the latest uv release..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi

  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return
  fi

  echo "uv installation failed" >&2
  exit 1
}

create_venv() {
  if [[ -d "${VENV_PATH}" ]]; then
    return
  fi

  echo "Creating virtual environment at ${VENV_PATH} using uv..."
  "${UV_BIN}" venv "${VENV_PATH}"
}

trim() {
  local var="$1"
  var="${var#"${var%%[![:space:]]*}"}"
  var="${var%"${var##*[![:space:]]}"}"
  printf '%s' "$var"
}

declare -a REQUIREMENT_OPTIONS=()
declare -a PYTORCH_CPU_LINES=()
declare -a GENERAL_REQUIREMENTS=()
INSTALLS_TORCH=0

collect_requirements() {
  local raw trimmed name lower

  while IFS= read -r raw || [[ -n "$raw" ]]; do
    trimmed="${raw%%#*}"
    trimmed="$(trim "$trimmed")"

    if [[ -z "$trimmed" ]]; then
      continue
    fi

    if [[ "$trimmed" == --* ]]; then
      REQUIREMENT_OPTIONS+=("$trimmed")
      continue
    fi

    name="${trimmed%%[<>=!~ ;[]*}"
    lower="${name,,}"

    case "$lower" in
      torch|torchvision|torchaudio)
        PYTORCH_CPU_LINES+=("$trimmed")
        if [[ "$lower" == "torch" ]]; then
          INSTALLS_TORCH=1
        fi
        ;;
      *)
        GENERAL_REQUIREMENTS+=("$trimmed")
        ;;
    esac
  done < "$1"
}

install_cpu_pytorch_packages() {
  if [[ ${#PYTORCH_CPU_LINES[@]} -eq 0 ]]; then
    return
  fi

  echo "Installing PyTorch packages from ${PYTORCH_CPU_INDEX} (CPU-only wheels)..."
  "${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" --index-url "${PYTORCH_CPU_INDEX}" "${PYTORCH_CPU_LINES[@]}"
}

install_remaining_requirements() {
  if [[ ${#GENERAL_REQUIREMENTS[@]} -eq 0 ]]; then
    return
  fi

  echo "Installing remaining dependencies from ${REQUIREMENTS_FILE} using uv..."
  "${UV_BIN}" pip install --python "${VENV_PATH}/bin/python" "${REQUIREMENT_OPTIONS[@]}" "${GENERAL_REQUIREMENTS[@]}"
}

verify_cpu_only_torch() {
  if [[ ${INSTALLS_TORCH} -eq 0 ]]; then
    return
  fi

  echo "Verifying that the installed torch build is CPU-only..."
  if ! "${VENV_PATH}/bin/python" - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive in shell script
    sys.exit(f"failed to import torch: {exc}")

if torch.cuda.is_available():
    sys.exit("CUDA appears to be available; expected CPU-only torch build")

cuda_version = getattr(torch.version, "cuda", None)
if cuda_version not in (None, "0.0"):
    sys.exit(f"CUDA version detected: {cuda_version}")
PY
  then
    echo "torch installation does not appear to be CPU-only." >&2
    exit 1
  fi
  echo "Confirmed torch CPU-only build."
}

collect_requirements "${REQUIREMENTS_FILE}"

ensure_uv
create_venv
install_cpu_pytorch_packages
install_remaining_requirements
verify_cpu_only_torch

echo
echo "Installation complete. Activate the environment with:"
echo "  source \"${VENV_PATH}/bin/activate\""
