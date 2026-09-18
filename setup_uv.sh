#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-output/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-output/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-output/.uv-python}"
UV_BOOTSTRAP_DIR="${UV_BOOTSTRAP_DIR:-output/uv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python}"

mkdir -p output

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed or not on PATH. Installing uv under ${UV_BOOTSTRAP_DIR}..."
    "$PYTHON_BOOTSTRAP" -m pip install --prefix "$UV_BOOTSTRAP_DIR" uv
    export PATH="$(pwd)/${UV_BOOTSTRAP_DIR}/bin:${PATH}"
fi

uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.11
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" -r requirements-uv.txt

echo
echo "Environment ready. Run:"
echo "source ${UV_PROJECT_ENVIRONMENT}/bin/activate"
echo "bash toy_example.sh"
