#!/usr/bin/env bash
# Download Piper voices into models/piper. Nothing here is committed.
# Usage: bash deploy/linux/download_voices.sh en_US-lessac-medium es_MX-ald-medium fr_FR-siwis-medium
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/models/piper"
mkdir -p "$DEST"
if [[ "$#" -lt 1 ]]; then
  echo "Pass Piper voice ids, for example en_US-lessac-medium en_GB-alba-medium es_ES-davefx-medium fr_FR-siwis-medium" >&2
  echo "There is no Piper voice for Urdu, Punjabi, Sindhi, or Haitian Creole." >&2
  exit 2
fi
for voice in "$@"; do
  lang="${voice%%-*}"
  family="${lang%%_*}"
  region="${lang#*_}"
  name="${voice#*-}"
  name="${name%-medium}"
  name="${name%-high}"
  name="${name%-low}"
  name="${name%-x_low}"
  quality="${voice##*-}"
  base="https://huggingface.co/rhasspy/piper-voices/resolve/main/${family}/${lang}/${name}/${quality}"
  echo "Fetching ${voice}"
  curl -fL --retry 3 -o "$DEST/${voice}.onnx" "${base}/${voice}.onnx"
  curl -fL --retry 3 -o "$DEST/${voice}.onnx.json" "${base}/${voice}.onnx.json"
done
echo "Voices are in $DEST"
