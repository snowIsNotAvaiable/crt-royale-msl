# RetroVisor Integration Files

The files in this directory are the **canonical, source-of-truth versions** of the CRT-Royale integration into the RetroVisor app:

- `CrtRoyale.metal` — the full MSL pipeline (~1560 lines, 12 passes + helpers)
- `CrtRoyale.swift` — Swift integration class (settings, kernel dispatch, texture management)
- `textures/TileableLinearApertureGrille15Wide8And5d5Spacing.png` — the aperture-grille mask LUT shipped from `vendor/slang-shaders/`
- `build-patches/*.patch` — small diffs to vendor's tracked files (Xcode project, Main.storyboard, ShaderLibrary.swift) so that RetroVisor knows about CRT-Royale at build time

## Why this directory exists

RetroVisor's upstream repo (`dirkwhoffmann/RetroVisor`) is read-only for us. We don't want to commit CRT-Royale-specific changes into it, but the files have to live at the paths Xcode expects:

```
vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal
vendor/RetroVisor/RetroVisor/Shaders/CrtRoyale.swift
vendor/RetroVisor/RetroVisor/Resources/TileableLinearApertureGrille15Wide8And5d5Spacing.png
```

Plus the Xcode project (`RetroVisor.xcodeproj/project.pbxproj`), `Main.storyboard`, and `ShaderLibrary.swift` need small additions to actually compile + link our new files.

**Solution:** the canonical files live here (committed in this repo); the vendor paths are **symlinks** pointing back to `crt-royale-msl/integration/`; the 3 patched build files in vendor are kept locally modified and **`skip-worktree`-flagged** so vendor's `git status` stays clean. The patches are versioned in `build-patches/` so the build-time edits are also stored in our own repo.

## Fresh-clone setup

After cloning this project on a new machine (or wiping `vendor/RetroVisor/`):

```bash
# macOS / Linux:
bash crt-royale-msl/integration/setup.sh

# Windows (PowerShell):
pwsh crt-royale-msl/integration/setup.ps1
```

Both wrappers call the cross-platform Python script `setup.py`, which:

1. Creates symlinks at the three vendor paths pointing back here (or **file copies** if symlinks aren't allowed -- Windows without Developer Mode).
2. Applies the build-integration patches to vendor's tracked files (idempotent: re-running detects already-applied state).
3. Flags the patched vendor files as `skip-worktree` so the local modifications don't show up in vendor's `git status`.
4. Adds the 3 untracked symlink paths to vendor's `.git/info/exclude` (inside a `# >>> crt-royale-msl integration` marker block so undo can clean up cleanly).
5. Verifies that vendor's working tree is git-clean and reports any unexpected residual changes.

After that, `git status` in `vendor/RetroVisor/` should be clean (`0 ahead, 0 behind, nothing to commit`).

### Force file copies instead of symlinks

```bash
python3 crt-royale-msl/integration/setup.py --copy
```

Use this on Windows without Developer Mode, or when symlinks cause issues with your editor / IDE. **Caveat:** with copies, editing a file in `integration/` doesn't automatically propagate to vendor -- you have to re-run `setup.py` to refresh the copies.

### Undo the setup

```bash
python3 crt-royale-msl/integration/setup.py --undo
```

Removes the symlinks/copies, reverses the patches (so vendor's tracked files are back to upstream content), clears the `skip-worktree` flags, and removes our entries from `.git/info/exclude`. Useful before doing a real upstream merge or `git pull` in vendor.

## Editing workflow

- Edit the file under `crt-royale-msl/integration/` (or `vendor/RetroVisor/...`, equivalent through the symlink on Mac/Linux).
- The change is immediately visible to Xcode through the symlink (no copy/sync step on Mac/Linux). On Windows with file copies, re-run `setup.py` to push your edits to vendor.
- Commit your change in **this** repo (`crt-royale-msl/`). Vendor's `git status` stays clean.

## Updating the build-integration patches

If you need to extend the integration (e.g. add a new MSL kernel that needs another Xcode build phase), edit the files in vendor temporarily (the `skip-worktree` flag has to be cleared first via `git update-index --no-skip-worktree <path>`), then regenerate the patch:

```bash
cd vendor/RetroVisor
git update-index --no-skip-worktree RetroVisor/Shaders/ShaderLibrary.swift
# ... make your edits ...
git diff HEAD -- RetroVisor/Shaders/ShaderLibrary.swift \
  > ../../crt-royale-msl/integration/build-patches/03-ShaderLibrary.swift.patch
git update-index --skip-worktree RetroVisor/Shaders/ShaderLibrary.swift
```

Then commit the updated `.patch` in your own repo.
