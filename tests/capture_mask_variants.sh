#!/usr/bin/env bash
# Rendert das colorbars-Pattern durch die MSL-Pipeline einmal pro Maskentyp
# (Aperture Grille / Slot / Shadow EDP) und legt die Snapshots unter
# tests/outputs/mask-<typ>/ ab.
#
# Warum ein eigenes Skript: validate.sh faehrt bewusst nur den Default-Pfad
# (Grille), weil die Sanity-Checks und die Reference-Vergleiche darauf
# kalibriert sind. Die Maskentyp-Abbildung im Bericht (fig-mask-types.png) und
# das GIF auf der Ergebnisseite brauchen dagegen alle drei Varianten. Vorher
# lagen die dafuer noetigen Renders in einem Scratch-Verzeichnis ausserhalb des
# Repos; damit war die Abbildung nicht reproduzierbar.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

METAL_SRC="${CRT_ROYALE_METAL:-$REPO_ROOT/integration/CrtRoyale.metal}"
INPUT="${CRT_ROYALE_MASK_INPUT:-$REPO_ROOT/tests/inputs/colorbars.png}"
SCALE="${CRT_ROYALE_SCALE:-4}"
LUT_DIR="${CRT_ROYALE_LUT_DIR:-$WORKSPACE_ROOT/vendor/slang-shaders/crt/shaders/crt-royale}"
RUNNER_DIR="$HERE/SwiftRunner"
RUNNER_BIN="$RUNNER_DIR/.build/release/SwiftRunner"

# (typ, grosse LUT, kleine LUT)
declare -a VARIANTS=(
  "grille:TileableLinearApertureGrille15Wide8And5d5Spacing:TileableLinearApertureGrille15Wide8And5d5SpacingResizeTo64"
  "slot:TileableLinearSlotMaskTall15Wide9And4d5Horizontal9d14VerticalSpacing:TileableLinearSlotMaskTall15Wide9And4d5Horizontal9d14VerticalSpacingResizeTo64"
  "shadow:TileableLinearShadowMaskEDP:TileableLinearShadowMaskEDPResizeTo64"
)

if [[ ! -x "$RUNNER_BIN" ]]; then
  echo "==> Baue SwiftRunner"
  (cd "$RUNNER_DIR" && swift build -c release)
fi

for v in "${VARIANTS[@]}"; do
  IFS=':' read -r type lut_large lut_small <<< "$v"
  lut_large_path="$LUT_DIR/$lut_large.png"
  lut_small_path="$LUT_DIR/$lut_small.png"
  outdir="$REPO_ROOT/tests/outputs/mask-$type"

  for p in "$lut_large_path" "$lut_small_path"; do
    if [[ ! -f "$p" ]]; then
      echo "FEHLER: LUT fehlt: $p" >&2
      exit 1
    fi
  done

  mkdir -p "$outdir"
  echo "==> mask-type $type"
  "$RUNNER_BIN" --metal "$METAL_SRC" --input "$INPUT" \
      --outdir "$outdir" --scale "$SCALE" \
      --mask-lut "$lut_large_path" --mask-lut-small "$lut_small_path" \
      --mask-type "$type" > /dev/null
done

echo
echo "Snapshots: $REPO_ROOT/tests/outputs/mask-{grille,slot,shadow}/"
echo "Abbildungen neu erzeugen: python3 tools/make_extra_figures.py"
