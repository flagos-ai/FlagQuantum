#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
INSTALL_JAX=0

usage() {
    cat <<'EOF'
Usage: tools/setup_flagquantum_dev_env.sh [--with-jax]

Creates a dedicated local virtual environment for FlagQuantum development at
.venv/ and installs the project in editable mode with development dependencies.

Environment variables:
  PYTHON_BIN   Python interpreter to use (default: python3)
  VENV_DIR     Virtualenv path (default: <repo>/.venv)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-jax)
            INSTALL_JAX=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import venv" >/dev/null 2>&1; then
    echo "The interpreter '${PYTHON_BIN}' does not provide the stdlib venv module." >&2
    echo "Install a full Python distribution first, then rerun this script." >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
        cat >&2 <<EOF
Failed to create a virtual environment with '${PYTHON_BIN}'.
This usually means the interpreter is missing ensurepip/venv support.
Install a full Python toolchain, then rerun:

  PYTHON_BIN=${PYTHON_BIN} ${SCRIPT_DIR}/setup_flagquantum_dev_env.sh
EOF
        exit 1
    fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}[dev]"

if [[ "${INSTALL_JAX}" -eq 1 ]]; then
    "${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}[jax]"
fi

cat <<EOF
FlagQuantum development environment is ready.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Sanity checks:
  python -c "import flagquantum as fq; print(fq.__version__)"
  python -m pytest -m "smoke or unit" -q
EOF
