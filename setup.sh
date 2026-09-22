#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/tff_env}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "No se encontró ${PYTHON_BIN}. Instala Python 3.11 o superior." >&2
    exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

# TFF 0.86.0 declara jaxlib 0.4.14, cuyo wheel no está disponible para Python 3.11.
"${VENV_DIR}/bin/python" -m pip install --no-deps "tensorflow-federated==0.86.0"

echo "Entorno preparado en ${VENV_DIR}"
echo "Actívalo con: source ${VENV_DIR}/bin/activate"
