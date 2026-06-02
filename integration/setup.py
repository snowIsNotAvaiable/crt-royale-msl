#!/usr/bin/env python3
"""Setup the CRT-Royale integration into vendor/RetroVisor on a fresh clone.

Cross-platform: works on macOS, Linux, and Windows (Windows requires Developer
Mode enabled for symlink creation -- alternatively this script falls back to
file copies on Windows if symlinks fail).

What it does:

1. Apply 3 build-integration patches to vendor's tracked files (Xcode project,
   Main.storyboard, ShaderLibrary.swift) so that RetroVisor knows about CRT-
   Royale and links it into the build.

2. Link or copy the 3 implementation files (CrtRoyale.metal, CrtRoyale.swift,
   the aperture-grille LUT PNG) from crt-royale-msl/integration/ into vendor's
   expected paths.

3. Mark vendor's tracked files as `skip-worktree` so the patched versions don't
   show up in `git status`. Add the new untracked files to `.git/info/exclude`.

After this script: vendor/RetroVisor/ working tree is git-clean, the CRT-Royale
shader is fully wired into the Xcode project, and editing the canonical files
in crt-royale-msl/integration/ is reflected immediately (when linked) or after
re-running this script (when copied).

Usage:
    python3 crt-royale-msl/integration/setup.py [--copy] [--undo]

Flags:
    --copy   Force file copies instead of symlinks (useful on Windows without
             Developer Mode). The copies become stale if integration/ changes,
             so re-run the script after each edit.
    --undo   Reverse the setup: remove symlinks/copies, drop skip-worktree
             flags, and remove our entries from .git/info/exclude.

The script is idempotent: running it twice does nothing harmful.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
VENDOR = ROOT / "vendor" / "RetroVisor"

# (canonical_relpath_under_integration, vendor_relpath_under_RetroVisor_dir).
SYMLINK_FILES = [
    ("CrtRoyale.metal",                                                       "RetroVisor/GPU/CrtRoyale.metal"),
    ("CrtRoyale.swift",                                                       "RetroVisor/Shaders/CrtRoyale.swift"),
    ("textures/TileableLinearApertureGrille15Wide8And5d5Spacing.png",         "RetroVisor/Resources/TileableLinearApertureGrille15Wide8And5d5Spacing.png"),
]

# Patches applied to upstream-tracked files. Order matters only loosely.
PATCH_FILES = [
    ("build-patches/01-project.pbxproj.patch",      "RetroVisor.xcodeproj/project.pbxproj"),
    ("build-patches/02-Main.storyboard.patch",      "RetroVisor/Base.lproj/Main.storyboard"),
    ("build-patches/03-ShaderLibrary.swift.patch",  "RetroVisor/Shaders/ShaderLibrary.swift"),
]

EXCLUDE_MARKER_BEGIN = "# >>> crt-royale-msl integration"
EXCLUDE_MARKER_END   = "# <<< crt-royale-msl integration"


def err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(msg)


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd),
                          capture_output=True, text=True)


def check_vendor() -> None:
    if not (VENDOR / ".git").exists():
        err(f"{VENDOR} is not a git repository.")
        err("Clone dirkwhoffmann/RetroVisor into vendor/ first.")
        sys.exit(1)


def make_symlink_or_copy(src: Path, dst: Path, force_copy: bool) -> str:
    """Create a symlink (preferred) or fall back to a file copy. Returns
    the mode actually used: "symlink", "copy", or "skip"."""
    if dst.is_symlink() or dst.exists():
        # If already correctly linked / identical, skip.
        if dst.is_symlink():
            try:
                resolved = dst.resolve(strict=False)
                if resolved == src.resolve():
                    return "skip"
            except OSError:
                pass
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        # Symlink with the relative target so it survives a moved project root.
        rel = os.path.relpath(src, dst.parent)
        try:
            os.symlink(rel, dst)
            return "symlink"
        except OSError as e:
            info(f"    symlink failed ({e}); falling back to copy")
    shutil.copy2(src, dst)
    return "copy"


def remove_link(dst: Path) -> bool:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
        return True
    return False


def apply_patch(patch_file: Path, target: Path, vendor_dir: Path) -> str:
    """Apply a unified-diff patch to a vendor-tracked file. Returns "applied",
    "already", or "failed"."""
    if not patch_file.exists() or patch_file.stat().st_size == 0:
        return "skip"
    # Our patches are plain `diff -u` output (no a/ b/ prefixes), so use -p0.
    base_args = ["git", "apply", "-p0", "--unsafe-paths",
                 "--directory=" + str(vendor_dir)]
    # Try reverse first to detect already-applied state.
    check = subprocess.run(
        [*base_args, "--check", "--reverse", str(patch_file)],
        capture_output=True, text=True
    )
    if check.returncode == 0:
        return "already"
    apply = subprocess.run(
        [*base_args, str(patch_file)],
        capture_output=True, text=True
    )
    if apply.returncode == 0:
        return "applied"
    info(f"    git apply stderr: {apply.stderr.strip()}")
    return "failed"


def reverse_patch(patch_file: Path, vendor_dir: Path) -> str:
    if not patch_file.exists() or patch_file.stat().st_size == 0:
        return "skip"
    apply = subprocess.run(
        ["git", "apply", "-p0", "--unsafe-paths",
         "--directory=" + str(vendor_dir),
         "--reverse", str(patch_file)],
        capture_output=True, text=True
    )
    return "reversed" if apply.returncode == 0 else "skip"


def set_skip_worktree(rel: str, on: bool, vendor_dir: Path) -> str:
    flag = "--skip-worktree" if on else "--no-skip-worktree"
    res = run_git(["update-index", flag, rel], cwd=vendor_dir)
    if res.returncode != 0:
        return "n/a"
    return "set" if on else "cleared"


def update_exclude(vendor_dir: Path, paths_to_add: list[str], undo: bool) -> None:
    exclude_file = vendor_dir / ".git" / "info" / "exclude"
    if not exclude_file.exists():
        exclude_file.write_text("")
    lines = exclude_file.read_text().splitlines()
    in_block = False
    cleaned = []
    for line in lines:
        if line == EXCLUDE_MARKER_BEGIN:
            in_block = True
            continue
        if line == EXCLUDE_MARKER_END:
            in_block = False
            continue
        if not in_block:
            cleaned.append(line)
    if not undo:
        cleaned.append(EXCLUDE_MARKER_BEGIN)
        cleaned.extend(paths_to_add)
        cleaned.append(EXCLUDE_MARKER_END)
    exclude_file.write_text("\n".join(cleaned).rstrip() + "\n")


def cmd_setup(force_copy: bool) -> int:
    check_vendor()
    info(f"==> Linking integration files into {VENDOR.name}/")
    for src_rel, dst_rel in SYMLINK_FILES:
        src = HERE / src_rel
        dst = VENDOR / dst_rel
        if not src.exists():
            info(f"    SKIP   {dst_rel} (no source at {src_rel})")
            continue
        mode = make_symlink_or_copy(src, dst, force_copy)
        info(f"    {mode:7s} {dst_rel}")

    info("")
    info("==> Applying build-integration patches")
    for patch_rel, target_rel in PATCH_FILES:
        patch = HERE / patch_rel
        status = apply_patch(patch, VENDOR / target_rel, VENDOR)
        info(f"    {status:8s} {target_rel}")

    info("")
    info("==> Marking patched files as skip-worktree")
    for _, target_rel in PATCH_FILES:
        res = run_git(["ls-files", "--error-unmatch", target_rel], cwd=VENDOR)
        if res.returncode == 0:
            set_skip_worktree(target_rel, True, VENDOR)
            info(f"    set     {target_rel}")
        else:
            info(f"    n/a     {target_rel} (not tracked in vendor)")

    info("")
    info("==> Updating vendor/.git/info/exclude")
    untracked_paths = [dst_rel for _, dst_rel in SYMLINK_FILES]
    update_exclude(VENDOR, untracked_paths, undo=False)
    info(f"    wrote {len(untracked_paths)} entries inside crt-royale-msl markers")

    info("")
    info("==> Verifying vendor git status")
    res = run_git(["status", "--short"], cwd=VENDOR)
    if res.stdout.strip():
        info("    WARN: vendor still has unexpected changes:")
        for line in res.stdout.strip().splitlines():
            info(f"      {line}")
        return 1
    info("    CLEAN")
    return 0


def cmd_undo() -> int:
    check_vendor()
    info(f"==> Removing integration symlinks/copies from {VENDOR.name}/")
    for _, dst_rel in SYMLINK_FILES:
        removed = remove_link(VENDOR / dst_rel)
        info(f"    {'removed' if removed else 'absent ':7s} {dst_rel}")

    info("")
    info("==> Reverting build-integration patches")
    for patch_rel, target_rel in PATCH_FILES:
        # Need to clear skip-worktree first so git can see the file.
        set_skip_worktree(target_rel, False, VENDOR)
        status = reverse_patch(HERE / patch_rel, VENDOR)
        info(f"    {status:8s} {target_rel}")

    info("")
    info("==> Cleaning vendor/.git/info/exclude")
    update_exclude(VENDOR, [], undo=True)
    info("    crt-royale-msl block removed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--copy", action="store_true",
                    help="Force file copies instead of symlinks (Windows fallback).")
    ap.add_argument("--undo", action="store_true",
                    help="Reverse the setup.")
    args = ap.parse_args()
    if args.undo:
        return cmd_undo()
    return cmd_setup(force_copy=args.copy)


if __name__ == "__main__":
    sys.exit(main())
