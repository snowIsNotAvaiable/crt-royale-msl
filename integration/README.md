# RetroVisor Integration Files

The files in this directory are the **canonical, source-of-truth versions** of the CRT-Royale integration into the RetroVisor app:

- `CrtRoyale.metal` — the full MSL pipeline (~1560 lines, 12 passes + helpers)
- `CrtRoyale.swift` — Swift integration class (settings, kernel dispatch, texture management)
- `textures/TileableLinearApertureGrille15Wide8And5d5Spacing.png` — the aperture-grille mask LUT shipped from `vendor/slang-shaders/`

## Why this directory exists

RetroVisor's upstream repo (`dirkwhoffmann/RetroVisor`) is read-only for us. We don't want to commit CRT-Royale-specific changes into it, but the files have to live at the paths Xcode expects:

```
vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal
vendor/RetroVisor/RetroVisor/Shaders/CrtRoyale.swift
vendor/RetroVisor/RetroVisor/Resources/TileableLinearApertureGrille15Wide8And5d5Spacing.png
```

Solution: the **canonical files live here** (committed in this repo), and the vendor paths are **symlinks** pointing back to `crt-royale-msl/integration/`. Editing a file here is automatically visible to Xcode through the symlinks.

To hide the resulting type-changed paths from vendor's git status:
- The 2 tracked files (`CrtRoyale.metal`, `CrtRoyale.swift`) are marked with `git update-index --skip-worktree` in the vendor repo.
- The untracked LUT symlink is listed in `vendor/RetroVisor/.git/info/exclude`.

## Fresh-clone setup

After cloning this project on a new machine (or wiping `vendor/RetroVisor/`):

```bash
bash crt-royale-msl/integration/setup-symlinks.sh
```

This script:
1. Removes any existing files at the three vendor paths.
2. Creates symlinks back to `crt-royale-msl/integration/`.
3. Applies `skip-worktree` to the 2 tracked vendor files.
4. Adds the LUT path to vendor's `.git/info/exclude`.

After that, `git status` in `vendor/RetroVisor/` should be clean.

## Editing workflow

- Edit the file under `crt-royale-msl/integration/` (or `vendor/RetroVisor/...`, equivalent through the symlink).
- The change is immediately visible to Xcode (no copy/sync step).
- Commit your change in **this** repo (`crt-royale-msl/`). Vendor's git status stays clean.

## Restoring vendor's original state (undo)

If you ever need to detach from this symlink setup (e.g. to do a real upstream rebase):

```bash
cd vendor/RetroVisor
git update-index --no-skip-worktree RetroVisor/GPU/CrtRoyale.metal RetroVisor/Shaders/CrtRoyale.swift
# Then `git checkout HEAD --` on each file to restore the real-file version
# committed in 81e29f4.
```
