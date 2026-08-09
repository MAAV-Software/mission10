#!/usr/bin/env bash
# Fetch the offline speech models for jarvis_web.
#
# These stay out of git: ~50 MB of vosk model plus ~60 MB of piper voice, and one
# fat commit poisons every clone. They land under models/speech/assets/, which the
# existing models/*/assets/ rule in .gitignore already covers.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO}/models/speech/assets"

VOSK_MODEL="vosk-model-small-en-us-0.15"
VOSK_URL="https://alphacephei.com/vosk/models/${VOSK_MODEL}.zip"

PIPER_VOICE="en_US-lessac-medium"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

mkdir -p "${DEST}"

if [ -d "${DEST}/${VOSK_MODEL}" ]; then
  echo "vosk model present, skipping"
else
  echo "fetching ${VOSK_MODEL}..."
  tmp="$(mktemp --suffix=.zip)"
  curl -fL "${VOSK_URL}" -o "${tmp}"
  unzip -q "${tmp}" -d "${DEST}"
  rm -f "${tmp}"
fi

# piper needs the weights and the sidecar config; the config alone is silent
# failure at load time, so fetch them as a pair.
for suffix in onnx onnx.json; do
  target="${DEST}/${PIPER_VOICE}.${suffix}"
  if [ -f "${target}" ]; then
    echo "piper ${suffix} present, skipping"
  else
    echo "fetching ${PIPER_VOICE}.${suffix}..."
    curl -fL "${PIPER_BASE}/${PIPER_VOICE}.${suffix}" -o "${target}"
  fi
done

echo "speech models in ${DEST}"
