#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/convert_demo_music.sh [--force]

Converts WAV/FLAC/MP3 files from assets/demo-music/generated/raw/
to normalized MP3 files in assets/demo-music/generated/.

Options:
  -f, --force   overwrite existing .mp3 outputs
  -h, --help    show this help
USAGE
}

force=0

while (($#)); do
  case "$1" in
    -f | --force)
      force=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
raw_dir="$repo_root/assets/demo-music/generated/raw"
out_dir="$repo_root/assets/demo-music/generated"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install ffmpeg with MP3/LAME support." >&2
  exit 1
fi

encoders="$(ffmpeg -hide_banner -encoders 2>&1)"
if [[ "$encoders" != *libmp3lame* ]]; then
  echo "ffmpeg does not report the libmp3lame encoder." >&2
  echo "Install ffmpeg with MP3/LAME support, or document an MP3-capable fallback in assets/demo-music/GENERATION.md." >&2
  exit 1
fi

if [[ ! -d "$raw_dir" ]]; then
  echo "Missing input directory: $raw_dir" >&2
  exit 1
fi

mkdir -p "$out_dir"

mapfile -d '' inputs < <(
  find "$raw_dir" -maxdepth 1 -type f \
    \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' \) \
    -print0 | sort -z
)

if ((${#inputs[@]} == 0)); then
  echo "No WAV/FLAC/MP3 files found in $raw_dir"
  exit 0
fi

converted=0
skipped=0

for input in "${inputs[@]}"; do
  filename="$(basename -- "$input")"
  stem="${filename%.*}"
  output="$out_dir/$stem.mp3"
  temp_output="$out_dir/.$stem.tmp.$$.mp3"

  if [[ -e "$output" && "$force" -ne 1 ]]; then
    echo "Skipping existing output: ${output#$repo_root/}"
    skipped=$((skipped + 1))
    continue
  fi

  echo "Converting ${input#$repo_root/} -> ${output#$repo_root/}"
  if ffmpeg -hide_banner -y \
    -i "$input" \
    -vn \
    -af loudnorm=I=-16:TP=-1.5:LRA=11 \
    -ar 48000 \
    -c:a libmp3lame \
    -q:a 3 \
    -id3v2_version 3 \
    "$temp_output"; then
    mv -f "$temp_output" "$output"
    converted=$((converted + 1))
  else
    rm -f "$temp_output"
    echo "Conversion failed for ${input#$repo_root/}" >&2
    exit 1
  fi
done

echo "Done. Converted: $converted. Skipped: $skipped."
