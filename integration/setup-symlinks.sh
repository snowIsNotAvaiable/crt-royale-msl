#!/usr/bin/env bash
# Wire the canonical integration files in crt-royale-msl/integration/ into the
# vendor/RetroVisor working tree as symlinks, and hide the resulting changes
# from vendor's git status (skip-worktree + .git/info/exclude).
#
# Run once after fresh checkout of the project. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENDOR="$ROOT/vendor/RetroVisor"

if [[ ! -d "$VENDOR/.git" ]]; then
    echo "FATAL: $VENDOR is not a git repository." >&2
    echo "Make sure you've cloned dirkwhoffmann/RetroVisor into vendor/ first." >&2
    exit 1
fi

# Pairs of "vendor relative path : integration relative path".
LINKS=(
    "RetroVisor/GPU/CrtRoyale.metal:CrtRoyale.metal"
    "RetroVisor/Shaders/CrtRoyale.swift:CrtRoyale.swift"
    "RetroVisor/Resources/TileableLinearApertureGrille15Wide8And5d5Spacing.png:textures/TileableLinearApertureGrille15Wide8And5d5Spacing.png"
)

echo "==> Creating symlinks vendor/ -> crt-royale-msl/integration/"
for pair in "${LINKS[@]}"; do
    IFS=':' read -r vendor_rel int_rel <<<"$pair"
    vendor_abs="$VENDOR/$vendor_rel"
    int_abs="$HERE/$int_rel"

    if [[ ! -f "$int_abs" ]]; then
        echo "  SKIP $vendor_rel (no source file at $int_rel)" >&2
        continue
    fi

    # Compute the relative target so the symlink works regardless of where
    # the project lives on disk.
    target="$(python3 -c "import os.path,sys; print(os.path.relpath(sys.argv[1], os.path.dirname(sys.argv[2])))" "$int_abs" "$vendor_abs")"

    # If a real file (not the right symlink) is there, replace it.
    if [[ -L "$vendor_abs" ]]; then
        current="$(readlink "$vendor_abs")"
        if [[ "$current" == "$target" ]]; then
            echo "  OK    $vendor_rel (already linked)"
            continue
        fi
        rm "$vendor_abs"
    elif [[ -e "$vendor_abs" ]]; then
        rm "$vendor_abs"
    fi
    ln -s "$target" "$vendor_abs"
    echo "  LINK  $vendor_rel -> $target"
done

echo
echo "==> Applying skip-worktree to tracked vendor files"
cd "$VENDOR"
for rel in "RetroVisor/GPU/CrtRoyale.metal" "RetroVisor/Shaders/CrtRoyale.swift"; do
    # Only skip-worktree if the path is tracked in vendor's index. If you
    # nuked the commit that adds them, this will silently fail -- that's ok.
    if git ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
        git update-index --skip-worktree "$rel"
        echo "  SKIP-WT $rel"
    else
        echo "  N/A     $rel (not tracked in vendor; nothing to hide)"
    fi
done

echo
echo "==> Adding untracked LUT symlink to vendor/.git/info/exclude"
EXCLUDE_FILE="$VENDOR/.git/info/exclude"
LUT_PATH="RetroVisor/Resources/TileableLinearApertureGrille15Wide8And5d5Spacing.png"
if grep -qxF "$LUT_PATH" "$EXCLUDE_FILE" 2>/dev/null; then
    echo "  OK    LUT already excluded"
else
    echo "$LUT_PATH" >> "$EXCLUDE_FILE"
    echo "  ADD   $LUT_PATH -> $EXCLUDE_FILE"
fi

echo
echo "==> Verifying vendor's git status is clean"
status_out="$(cd "$VENDOR" && git status --short)"
if [[ -z "$status_out" ]]; then
    echo "  CLEAN: vendor working tree has no uncommitted changes."
else
    echo "  WARN: vendor still has changes:" >&2
    echo "$status_out" | sed 's/^/    /' >&2
fi
echo
echo "Setup done."
