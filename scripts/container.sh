#!/usr/bin/env bash
set -euo pipefail

source container.env

CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity)"

if [[ ! -f "$UNBRANDING_IMAGE" ]]; then
  echo "container image not found: $UNBRANDING_IMAGE" >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

exec "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$UNBRANDING_ROOT" \
  --pwd "$UNBRANDING_REPO" \
  --env-file container.env \
  --env "UNBRANDING_GIT_COMMIT=$(git rev-parse --short HEAD)" \
  "$UNBRANDING_IMAGE" python3 "$@"
