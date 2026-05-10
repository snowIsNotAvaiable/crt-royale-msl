#!/usr/bin/env bash
# capture_reference.sh -- Run the original Slang crt-royale.slangp through
# librashader-cli for each test input, producing per-pass reference PNGs
# that line up 1:1 with our MSL pipeline outputs in tests/outputs/<mode>/.
#
# Output layout matches our outputs/ tree so tools/compare.py can iterate
# both side-by-side:
#   tests/reference/<pattern>/
#       01-pass1.png   # after Slang pass 0  (linearize)              -> our pass1
#       02-pass2.png   # after Slang pass 1  (vertical scanlines)     -> our pass2
#       03-pass3.png   # after Slang pass 7  (apply mask, simplified) -> our pass3
#       04-final.png   # after all 12 Slang passes
#
# Per-pass capture uses librashader-cli's --passes-enabled flag.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRESET="$ROOT/vendor/slang-shaders/crt/crt-royale.slangp"
INPUTS_DIR="$ROOT/crt-royale-msl/tests/inputs"
REF_DIR="$ROOT/crt-royale-msl/tests/reference"
LIBRASHADER="${LIBRASHADER_CLI:-$HOME/.cargo/bin/librashader-cli}"

# Output dimensions match our SwiftRunner (4x Y-upscale of 256x192 source).
DIM="256x768"

if [[ ! -x "$LIBRASHADER" ]]; then
  echo "librashader-cli not found at $LIBRASHADER" >&2
  echo "Install with: cargo +nightly install librashader-cli --no-default-features --features metal" >&2
  exit 1
fi

if [[ ! -f "$PRESET" ]]; then
  echo "preset not found: $PRESET" >&2
  exit 1
fi

# Slang preset has 12 passes (indexes 0..11). Stages we capture for comparison:
#   passes-enabled=1  -> after pass 0 (linearize) ~ our pass1
#   passes-enabled=2  -> after pass 1 (vertical scanlines) ~ our pass2
#   passes-enabled=8  -> after pass 7 (apply mask) ~ our pass3 (simplified)
#   passes-enabled=12 -> full pipeline = final
# These are the best-aligned stages between our reduced pipeline and the
# Slang reference; intermediate Slang passes (2..6, 8..10) have no MSL
# counterpart yet.
declare -a STAGES=(
  "01-pass1.png:1"
  "02-pass2.png:2"
  "03-pass3.png:8"
  "04-final.png:12"
)

mkdir -p "$REF_DIR"

INPUTS=("$INPUTS_DIR"/*.png)
if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "no inputs in $INPUTS_DIR" >&2
  exit 1
fi

echo "librashader: $("$LIBRASHADER" --version 2>/dev/null || echo unknown)"
echo "preset:     $PRESET"
echo "outputs:    $REF_DIR/<pattern>/"
echo

for input in "${INPUTS[@]}"; do
  name="$(basename "$input" .png)"
  out_dir="$REF_DIR/$name"
  mkdir -p "$out_dir"
  cp "$input" "$out_dir/00-input.png"
  echo "[$name]"
  for stage in "${STAGES[@]}"; do
    out_name="${stage%%:*}"
    n_passes="${stage##*:}"
    out_path="$out_dir/$out_name"
    "$LIBRASHADER" render \
      --preset "$PRESET" \
      --image "$input" \
      --out "$out_path" \
      --dimensions "$DIM" \
      --passes-enabled "$n_passes" \
      --runtime metal >/dev/null 2>&1
    echo "    $out_name (passes=$n_passes)"
  done
done

echo
echo "Done. Compare with our outputs:"
echo "  python3 $ROOT/crt-royale-msl/tools/compare.py \\"
echo "      --left  $ROOT/crt-royale-msl/tests/outputs/default/colorbars/04-final.png \\"
echo "      --right $REF_DIR/colorbars/04-final.png"
