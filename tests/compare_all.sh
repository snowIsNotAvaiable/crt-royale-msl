#!/usr/bin/env bash
# compare_all.sh -- Run tools/compare.py over every (pattern, pass) tuple
# against the librashader reference snapshots and print a summary table.
#
# Layout assumed:
#   tests/outputs/<mode>/<pattern>/{01-pass1,02-pass2,03-pass3,04-final}.png
#   tests/reference/<pattern>/{01-pass1,02-pass2,03-pass3,04-final}.png

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-default}"
OUT_DIR="$ROOT/crt-royale-msl/tests/outputs/$MODE"
REF_DIR="$ROOT/crt-royale-msl/tests/reference"
DIFF_DIR="$ROOT/crt-royale-msl/tests/diff/$MODE"
COMPARE="$ROOT/crt-royale-msl/tools/compare.py"

# Projekt-venv bevorzugen (siehe requirements.txt), sonst System-python3.
# validate.sh reicht seine Wahl via CRT_ROYALE_PY durch.
PY="${CRT_ROYALE_PY:-$ROOT/crt-royale-msl/.venv/bin/python3}"
[[ -x "$PY" ]] || PY="python3"

if [[ ! -d "$OUT_DIR" ]]; then
  echo "outputs not found: $OUT_DIR" >&2
  echo "run tests/validate.sh first" >&2
  exit 1
fi
if [[ ! -d "$REF_DIR" ]]; then
  echo "references not found: $REF_DIR" >&2
  echo "run tests/capture_reference.sh first" >&2
  exit 1
fi

mkdir -p "$DIFF_DIR"

# 01-pass1 ist bewusst ausgenommen: librashader exportiert Pass 1 in
# Viewport-Dimension (256x768), unser Snapshot in Quell-Dimension (256x192) --
# die Bilder sind per Design nicht pixelvergleichbar (siehe Bericht, Problem 6).
PASSES=(02-pass2 02b-bloom_approx 02c-halation_v 02d-halation_blur 02e-mask_resize_v 02f-mask_resize 03-pass3 03b-brightpass 03c-bloom_v 03d-bloom_final 04-final)

printf "%-14s %-10s %8s %8s %8s %8s\n" "pattern" "pass" "deltaE" "p95" "max" "ssim"
printf '%.0s-' {1..62}; echo

compare_failures=0

for pattern_dir in "$OUT_DIR"/*/; do
  pattern="$(basename "$pattern_dir")"
  ref_pattern_dir="$REF_DIR/$pattern"
  [[ -d "$ref_pattern_dir" ]] || continue
  diff_pattern_dir="$DIFF_DIR/$pattern"
  mkdir -p "$diff_pattern_dir"

  for p in "${PASSES[@]}"; do
    left="$pattern_dir$p.png"
    right="$ref_pattern_dir/$p.png"
    [[ -f "$left" && -f "$right" ]] || continue

    json_path="$diff_pattern_dir/$p.json"
    heat_path="$diff_pattern_dir/$p-heat.png"

    # Fehler nicht stumm schlucken: schlaegt compare.py fehl (z.B. fehlende
    # Python-Dependency), wuerde sonst eine STALE JSON aus einem frueheren
    # Lauf als aktuelles Ergebnis ausgegeben.
    if ! compare_err=$("$PY" "$COMPARE" \
      --left "$left" --right "$right" \
      --json "$json_path" --heatmap "$heat_path" \
      --threshold 99999 --quiet 2>&1 >/dev/null); then
      printf "%-14s %-10s %8s\n" "$pattern" "$p" "FEHLER"
      echo "  compare.py: ${compare_err:-unbekannter Fehler}" >&2
      compare_failures=$((compare_failures + 1))
      continue
    fi

    if [[ -f "$json_path" ]]; then
      mean=$("$PY" -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['mean']:.2f}\")")
      p95=$("$PY" -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['p95']:.2f}\")")
      mx=$("$PY" -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['max']:.2f}\")")
      ssim=$("$PY" -c "import json; d=json.load(open('$json_path')); print(f\"{d['ssim']:.3f}\")")
      printf "%-14s %-10s %8s %8s %8s %8s\n" "$pattern" "$p" "$mean" "$p95" "$mx" "$ssim"
    fi
  done
done

if (( compare_failures > 0 )); then
  echo
  echo "WARNUNG: $compare_failures compare.py-Aufrufe fehlgeschlagen -- Tabelle unvollstaendig." >&2
  echo "         Dependencies installieren:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

echo
echo "Heatmaps + JSON: $DIFF_DIR/<pattern>/"
