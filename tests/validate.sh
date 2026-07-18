#!/usr/bin/env bash
# CRT-Royale MSL validation pipeline.
#   1. (re)generates the deterministic test inputs
#   2. builds the headless Swift runner
#   3. runs it twice per input -- neutral gamma (round-trip identity) +
#      default gamma (gamma simulation visible) -- with 4x Y-upscale so
#      pass-2 scanline structure is actually visible
#   4. runs the Python analyzer against both result trees
#
# Run from anywhere; this script normalizes paths.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

METAL_SRC="${CRT_ROYALE_METAL_SRC:-$WORKSPACE_ROOT/vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal}"
MASK_LUT="${CRT_ROYALE_MASK_LUT:-$WORKSPACE_ROOT/vendor/slang-shaders/crt/shaders/crt-royale/TileableLinearApertureGrille15Wide8And5d5Spacing.png}"
MASK_LUT_SMALL="${CRT_ROYALE_MASK_LUT_SMALL:-$WORKSPACE_ROOT/vendor/slang-shaders/crt/shaders/crt-royale/TileableLinearApertureGrille15Wide8And5d5SpacingResizeTo64.png}"
INPUTS_DIR="$HERE/inputs"
OUT_NEUTRAL="$HERE/outputs/neutral"
OUT_DEFAULT="$HERE/outputs/default"
RUNNER_DIR="$HERE/SwiftRunner"
SCALE="${CRT_ROYALE_SCALE:-4}"

# Projekt-venv bevorzugen (siehe requirements.txt), sonst System-python3.
PY="$REPO_ROOT/.venv/bin/python3"
[[ -x "$PY" ]] || PY="python3"
export CRT_ROYALE_PY="$PY"

if [[ ! -f "$METAL_SRC" ]]; then
    echo "FATAL: cannot find CrtRoyale.metal at $METAL_SRC" >&2
    echo "       set CRT_ROYALE_METAL_SRC env var to override." >&2
    exit 2
fi

echo "==> [1/5] Generating test inputs"
"$PY" "$INPUTS_DIR/generate.py"

echo
echo "==> [2/5] Building Swift runner"
( cd "$RUNNER_DIR" && swift build -c release 2>&1 | tail -10 )
RUNNER_BIN="$RUNNER_DIR/.build/release/SwiftRunner"
if [[ ! -x "$RUNNER_BIN" ]]; then
    echo "FATAL: runner binary not found at $RUNNER_BIN" >&2
    exit 3
fi

run_suite() {
    local mode="$1"
    local out_root="$2"
    local -a extra_args=()
    if [[ -n "$3" ]]; then extra_args+=("$3"); fi
    if [[ -f "$MASK_LUT" ]]; then
        extra_args+=("--mask-lut" "$MASK_LUT")
    fi
    if [[ -f "$MASK_LUT_SMALL" ]]; then
        extra_args+=("--mask-lut-small" "$MASK_LUT_SMALL")
    fi
    rm -rf "$out_root"
    mkdir -p "$out_root"
    shopt -s nullglob
    for png in "$INPUTS_DIR"/*.png; do
        local name; name="$(basename "$png" .png)"
        "$RUNNER_BIN" --metal "$METAL_SRC" --input "$png" \
            --outdir "$out_root/$name" --scale "$SCALE" "${extra_args[@]}" \
            > /dev/null
    done
    echo "  wrote $out_root (mode=$mode, scale=${SCALE}x)"
}

echo
echo "==> [3/5] Running pipeline (neutral gamma, scale=${SCALE}x)"
run_suite neutral "$OUT_NEUTRAL" "--neutral"

echo
echo "==> [4/5] Running pipeline (default gamma, scale=${SCALE}x)"
run_suite default "$OUT_DEFAULT" ""

echo
echo "==> [5/5] Analyzing outputs"
neutral_status=0; default_status=0
"$PY" "$REPO_ROOT/tools/analyze.py" "$OUT_NEUTRAL" --mode neutral || neutral_status=$?
echo
"$PY" "$REPO_ROOT/tools/analyze.py" "$OUT_DEFAULT" --mode default || default_status=$?

if (( neutral_status != 0 || default_status != 0 )); then
    echo
    echo "FAIL: validation reported failures (neutral=$neutral_status, default=$default_status)"
    exit 1
fi
echo
echo "OK: all checks passed."

# Optional reference-comparison stage: only runs if librashader-cli is
# available AND a reference snapshot tree already exists. Reference snapshots
# are produced separately via tests/capture_reference.sh; we don't regenerate
# them on every validate.sh run (slow + Metal device-bound).
LIBRASHADER="${LIBRASHADER_CLI:-$HOME/.cargo/bin/librashader-cli}"
REF_DIR="$HERE/reference"
if [[ -x "$LIBRASHADER" && -d "$REF_DIR" ]]; then
    echo
    echo "==> [6/6] Comparing default outputs against librashader reference"
    "$HERE/compare_all.sh" default || true
fi
