#!/usr/bin/env bash
set -euo pipefail

source container.env

CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity)"

mkdir -p "$(dirname "$UNBRANDING_IMAGE")"

if [[ -f "$UNBRANDING_IMAGE" ]]; then
  echo "already present: $UNBRANDING_IMAGE"
  exit 0
fi

"$CONTAINER_RUNTIME" pull \
  "$UNBRANDING_IMAGE" \
  "docker://vllm/vllm-openai:$VLLM_VERSION"

"$CONTAINER_RUNTIME" exec --cleanenv "$UNBRANDING_IMAGE" python3 -c "
from importlib.metadata import version
for package in (\"vllm\", \"torch\", \"transformers\", \"pyyaml\"):
    print(f\"{package:14}\", version(package))
"
