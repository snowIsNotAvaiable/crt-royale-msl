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

PASSES=(01-pass1 02-pass2 02b-bloom_approx 02c-halation_v 02d-halation_blur 02e-mask_resize_v 02f-mask_resize 03-pass3 03b-brightpass 03c-bloom_v 03d-bloom_final 04-final)

printf "%-14s %-10s %8s %8s %8s %8s\n" "pattern" "pass" "deltaE" "p95" "max" "ssim"
printf '%.0s-' {1..62}; echo

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

    python3 "$COMPARE" \
      --left "$left" --right "$right" \
      --json "$json_path" --heatmap "$heat_path" \
      --threshold 99999 --quiet >/dev/null 2>&1 || true

    if [[ -f "$json_path" ]]; then
      mean=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['mean']:.2f}\")")
      p95=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['p95']:.2f}\")")
      mx=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['delta_e_2000']['max']:.2f}\")")
      ssim=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['ssim']:.3f}\")")
      printf "%-14s %-10s %8s %8s %8s %8s\n" "$pattern" "$p" "$mean" "$p95" "$mx" "$ssim"
    fi
  done
done

echo
echo "Heatmaps + JSON: $DIFF_DIR/<pattern>/"
