#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/container.env"

if [[ "$UNBRANDING_IMAGE" = /* ]]; then
  IMAGE="$UNBRANDING_IMAGE"
else
  IMAGE="$REPO_ROOT/$UNBRANDING_IMAGE"
fi

CONTAINER_RUNTIME="$(command -v apptainer || command -v singularity)"

mkdir -p "$(dirname "$IMAGE")"

if [[ -f "$IMAGE" ]]; then
  echo "already present: $IMAGE"
  exit 0
fi

"$CONTAINER_RUNTIME" pull \
  "$IMAGE" \
  "docker://vllm/vllm-openai:$VLLM_VERSION"

"$CONTAINER_RUNTIME" exec --cleanenv "$IMAGE" python3 -c "
from importlib.metadata import version
for package in (\"vllm\", \"torch\", \"transformers\", \"pyyaml\"):
    print(f\"{package:14}\", version(package))
"
