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
#   passes-enabled=3  -> after pass 2 (BLOOM_APPROX, 320x240) ~ our pass_bloom_approx
#   passes-enabled=4  -> after pass 3 (halation V, 320x240)   ~ our pass_halation_v
#   passes-enabled=5  -> after pass 4 (HALATION_BLUR, 320x240) ~ our pass_halation_h
#   passes-enabled=8  -> after pass 7 (apply mask) ~ our pass3
#   passes-enabled=9  -> after pass 8 (BRIGHTPASS) ~ our pass_brightpass
#   passes-enabled=10 -> after pass 9 (BLOOM_V)    ~ our pass_bloom_v
#   passes-enabled=11 -> after pass 10 (BLOOM_FINAL) ~ our pass_bloom_h_reconstitute
#   passes-enabled=12 -> full pipeline = final
# Stage format: name:n_passes:dim_override_or_dash:extra_params_or_dash
# Pass 5 (mask_resize_v) output dim depends on the rendered viewport.
# At our 256x768 default, pass 5 produces 64x48; pass 6 (MASK_RESIZE) is
# 16x48. We pin those via --dimensions so librashader doesn't blit-resize
# the per-pass snapshot before exporting.
declare -a STAGES=(
  "01-pass1.png:1:-:-"
  "02-pass2.png:2:-:-"
  "02b-bloom_approx.png:3:320x240:-"
  "02c-halation_v.png:4:320x240:-"
  "02d-halation_blur.png:5:320x240:-"
  "02e-mask_resize_v.png:6:64x48:-"
  "02f-mask_resize.png:7:16x48:-"
  "03-pass3.png:8:-:-"
  "03b-brightpass.png:9:-:-"
  "03c-bloom_v.png:10:-:-"
  "03d-bloom_final.png:11:-:-"
  "04-final.png:12:-:-"
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
    IFS=':' read -r out_name n_passes dim_override extra_params <<< "$stage"
    out_path="$out_dir/$out_name"
    stage_dim="$DIM"
    [[ "$dim_override" != "-" ]] && stage_dim="$dim_override"
    # mask_sample_mode_desired=1 forces the hardware-resample branch of pass
    # 7 (sample the large LUT directly via GPU mipmap+anisotropy). This is
    # our currently-validated default. Slang Pass 5+6 (Mask-Resize V/H) are
    # implemented in MSL but the discard-tile geometry isn't bit-exact yet,
    # so the mode-0 path (sampling MASK_RESIZE) deviates structurally from
    # librashader's reference. Keep the override here until that's tuned.
    params="mask_sample_mode_desired=1"
    [[ "$extra_params" != "-" ]] && params="$params,$extra_params"
    "$LIBRASHADER" render \
      --preset "$PRESET" \
      --image "$input" \
      --out "$out_path" \
      --dimensions "$stage_dim" \
      --passes-enabled "$n_passes" \
      --params "$params" \
      --runtime metal >/dev/null 2>&1
    echo "    $out_name (passes=$n_passes, dim=$stage_dim)"
  done
done

echo
echo "Done. Compare with our outputs:"
echo "  python3 $ROOT/crt-royale-msl/tools/compare.py \\"
echo "      --left  $ROOT/crt-royale-msl/tests/outputs/default/colorbars/04-final.png \\"
echo "      --right $REF_DIR/colorbars/04-final.png"
