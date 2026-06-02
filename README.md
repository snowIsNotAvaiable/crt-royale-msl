# CRT-Royale MSL Port for RetroVisor

Port of the [CRT-Royale](https://github.com/libretro/slang-shaders/tree/master/crt/shaders/crt-royale) shader (originally Slang/GLSL by TroggleMonkey, GPL v2+) to Apple Metal Shading Language (MSL), integrated into the [RetroVisor](https://github.com/dirkwhoffmann/RetroVisor) macOS app by Prof. Dr. Dirk W. Hoffmann.

**Master's project, HKA (Hochschule Karlsruhe).**
Devin Uyan — Software-Projekt, 1. Semester Master CS.

## Pipeline-Status

12 of 12 Slang passes ported. 11 of 12 ΔE-validated against the reference (`librashader-cli` running the original Slang pipeline). Default-pipeline runs at ~1050 FPS on Apple M2 at 256×768.

| Slang Pass | Beschreibung | Status |
|------------|--------------|--------|
| 0 | Linearize CRT Gamma + Bob | Portiert + validiert |
| 1 | Vertical Scanlines (Beam-Distribution) | Portiert + validiert |
| 2 | Bloom Approx (4×4 Gauss) | Portiert + validiert |
| 3+4 | Halation V/H (9-tap Gauss) | Portiert + validiert |
| 5+6 | Mask Resize V/H (Lanczos) | Portiert; Mode-0-Pfad nicht bit-exakt (Default ist Mode-1) |
| 7 | Apply Mask + Halation | Portiert + validiert |
| 8 | Brightpass (Area-Bloom-Extract) | Portiert + validiert |
| 9+10 | Bloom V/H + Reconstitute | Portiert + validiert |
| 11 | Geometry + AA + Final Encode | Portiert (Sphere-Raycaster + 16-tap Catmull-Rom-AA) |

## Verzeichnislayout

Dieses Repo (`crt-royale-msl`) liegt erwartungsgemäß **neben** zwei externen Repos, die getrennt geklont werden:

```
some-workspace/
├── crt-royale-msl/         ← dieses Repo
│   ├── integration/        # Source-of-Truth für die RetroVisor-Integration
│   │   ├── CrtRoyale.metal       # die produktive MSL-Pipeline (~1560 Zeilen)
│   │   ├── CrtRoyale.swift       # Swift-Integration (Settings, Kernels, Pipeline)
│   │   ├── textures/             # Phosphor-Mask LUTs
│   │   ├── build-patches/        # Patches für vendor's tracked Files (pbxproj etc.)
│   │   ├── setup.py              # cross-platform Setup (macOS/Linux/Windows)
│   │   ├── setup.sh              # bash-Wrapper
│   │   ├── setup.ps1             # PowerShell-Wrapper für Windows
│   │   └── README.md             # Detail-Erklärung des Setup-Mechanismus
│   ├── src/                # Studienmirror-Files je Pass (Doku-Mirror, nicht gebaut)
│   ├── tests/
│   │   ├── inputs/               # Procedural Test-Patterns (Color Bars, Grid, etc.)
│   │   ├── outputs/              # SwiftRunner Output-Snapshots
│   │   ├── reference/            # librashader-cli Reference-Snapshots
│   │   ├── diff/                 # ΔE2000 Heatmaps + Stats
│   │   ├── SwiftRunner/          # Headless Validation Harness (Swift Package)
│   │   ├── validate.sh           # Haupteinstiegspunkt für Tests
│   │   ├── capture_reference.sh  # Reference-Snapshots neu erzeugen
│   │   └── compare_all.sh        # Batch-Vergleich Outputs vs. Reference
│   ├── tools/
│   │   ├── compare.py            # ΔE2000 + SSIM + Heatmap
│   │   └── analyze.py            # Sanity-Checks für SwiftRunner-Outputs
│   ├── textures/                 # Mask-LUT-Working-Copies
│   └── docs/                     # NICHT versioniert -- privat (Bericht, Status etc.)
├── vendor/
│   ├── RetroVisor/         ← `git clone git@github.com:dirkwhoffmann/RetroVisor.git`
│   └── slang-shaders/      ← `git clone git@github.com:libretro/slang-shaders.git`
```

## Erste Schritte

### Voraussetzungen

- macOS 13+ (Apple Silicon getestet; Intel sollte funktionieren) -- oder Linux für die Tests (Build der App nur macOS).
- Xcode 15+ (für `swift build` und MSL-Compilation auf macOS).
- Python 3.9+ mit `numpy`, `Pillow`, `scikit-image` (für `compare.py` und `analyze.py`).
- Rust + librashader-cli (optional, für Reference-Snapshots).

### 1) Vendor-Repos klonen (parallel zu diesem Repo)

```bash
cd ..   # raus aus crt-royale-msl/
mkdir -p vendor && cd vendor
git clone git@github.com:dirkwhoffmann/RetroVisor.git
git clone git@github.com:libretro/slang-shaders.git
cd ..
```

### 2) Integration-Setup ausführen

Wickelt unsere `CrtRoyale.metal` / `CrtRoyale.swift` / LUT als Symlinks in den `vendor/RetroVisor`-Pfad und appliziert kleine Patches an Xcode-Projekt + Storyboard + ShaderLibrary. Details siehe [`integration/README.md`](integration/README.md).

```bash
# macOS / Linux:
bash crt-royale-msl/integration/setup.sh

# Windows (PowerShell, Python im PATH):
pwsh crt-royale-msl/integration/setup.ps1
```

Idempotent (mehrfaches Ausführen schadet nicht). Mit `--undo` voll rückgängig machbar.
Auf Windows ohne Developer-Mode automatisch Fallback auf File-Copies (oder explizit `--copy`).

### 3) RetroVisor App bauen

```bash
cd vendor/RetroVisor
open RetroVisor.xcodeproj
# ⌘B, dann ⌘R
```

In der laufenden App: Shader-Picker → **CRT-Royale**.

## Headless Validation

```bash
bash crt-royale-msl/tests/validate.sh
```

Was passiert:
1. Generiert deterministische Test-Inputs (Color Bars, Gradient, Grid, etc.).
2. Baut den Swift-Runner.
3. Fährt die ganze Pipeline für jedes Input-PNG durch (12 Slang-Passes + Geometry+AA).
4. Vergleicht den Output gegen die librashader-Reference-Snapshots in `tests/reference/`.

Erwartetes Resultat: **75 / 75 Sanity-Checks grün**; ΔE-Tabelle gegen Slang-Reference mit `04-final` Mean ΔE ≤ 5.04.

### Reference-Snapshots neu erzeugen (selten)

Wenn sich an der Slang-Source etwas ändert oder librashader-cli aktualisiert wird, kann die Reference neu generiert werden:

```bash
bash crt-royale-msl/tests/capture_reference.sh
```

Benötigt `librashader-cli` im PATH (Setup-Anleitung im internen `docs/Reference-Pipeline.md`).

### Performance-Benchmark

```bash
RUNNER=crt-royale-msl/tests/SwiftRunner/.build/release/SwiftRunner
"$RUNNER" \
    --metal vendor/RetroVisor/RetroVisor/GPU/CrtRoyale.metal \
    --input crt-royale-msl/tests/inputs/colorbars.png \
    --outdir /tmp/bench --bench 100 \
    --mask-lut vendor/slang-shaders/crt/shaders/crt-royale/TileableLinearApertureGrille15Wide8And5d5Spacing.png \
    --mask-lut-small vendor/slang-shaders/crt/shaders/crt-royale/TileableLinearApertureGrille15Wide8And5d5SpacingResizeTo64.png
```

Erwartete Ausgabe auf Apple M2: `p50 ≈ 0.95 ms` (~1050 FPS) bei 256×768.

## Methodik

Validierung findet in zwei Stufen statt:

1. **Sanity-Checks** (`tools/analyze.py`) -- structural checks am SwiftRunner-Output: Pass-1-Round-Trip-Identität bei `--neutral`, Y-Upscaling-Korrektheit, Range-Constraints, RGB-Triade-Konsistenz auf Solids.
2. **ΔE2000 + SSIM vs. Slang-Reference** (`tools/compare.py`) -- pixelgenauer Vergleich gegen librashader-cli-Output. Akzeptanzkriterium für portierte Stages: `mean ΔE2000 < 2.0`.

Beide Tools werden in `validate.sh` orchestriert. Die SwiftRunner-Pipeline produziert pro Test-Input 13 Snapshots (`00-input` … `04-final`, plus alle Zwischenstufen `02b-bloom_approx`, `02c-halation_v`, `02d-halation_blur`, `02e-mask_resize_v`, `02f-mask_resize`, `03-pass3`, `03b-brightpass`, `03c-bloom_v`, `03d-bloom_final`), so dass jeder Slang-Pass isoliert vergleichbar ist.

## Bekannte Limitationen

- **Mask-Resize Mode-0** (Lanczos-resized Mask-LUT) ist implementiert, aber die `mask_resize_tile_size`-Konstantenkette in librashader weicht von unserer Hand-Trace ab -- Default-Pipeline läuft im Mode-1-Pfad (hardware-resample, ΔE-validiert).
- **Curvature-Branch** in Pass 11 nicht ΔE-validiert (benötigt eigene Reference-Snapshots).
- **Geometrie-Modi 2+3** (Sphere_Alt, Cylinder) nicht portiert. Sphere (Mode 1) deckt den üblichen Curved-CRT-Fall.
- **AA-Filter-Konfigurabilität**: Catmull-Rom-Cubic fixiert (Slang exponiert 10 × 11 Filter/Sample-Matrix). Visuell äquivalent für die Moire-Suppression.

## Lizenzen

- Original CRT-Royale: GPL v2+ © 2014 TroggleMonkey.
- Diese Portierung: GPL v2+ (abgeleitetes Werk).
- RetroVisor: GPL v3 © Dirk W. Hoffmann.
