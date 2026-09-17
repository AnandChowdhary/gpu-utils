#!/usr/bin/env bash
# Renders an explainer for one package.
#   bash video/render.sh <package> [draft]
# Expects video/<package>_pipeline.py defining a Scene subclass named in PascalCase
# (e.g. gpu-view -> GpuViewPipeline). Output: video/media/videos/<package>_pipeline/1080p30/*.mp4
set -euo pipefail
cd "$(dirname "$0")"
pkg="${1:?package name required}"
mode="${2:-final}"
snake="${pkg//-/_}"
scene="$(echo "$pkg" | sed -E 's/(^|-)([a-z])/\U\2/g')Pipeline"

if ! [ -x .venv/bin/python ]; then
  uv venv .venv --python 3.12
fi
uv pip install --quiet --python .venv/bin/python -r requirements.txt

if [ "$mode" = draft ]; then
  .venv/bin/manim -ql "${snake}_pipeline.py" "$scene"
else
  .venv/bin/manim -qh --fps 30 --disable_caching "${snake}_pipeline.py" "$scene"
  out="media/videos/${snake}_pipeline/1080p30/${scene}.mp4"
  if command -v ffprobe >/dev/null; then
    d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
    echo "rendered $out (${d}s)"
    awk -v d="$d" 'BEGIN { if (d < 30 || d > 75) { print "duration out of 30-75s range"; exit 1 } }'
  fi
  mkdir -p "../apps/website/public"
  cp "$out" "../apps/website/public/${pkg}-pipeline.mp4"
fi
