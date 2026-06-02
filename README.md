# CRT-Royale MSL-Port für RetroVisor

Portierung des [CRT-Royale](https://github.com/libretro/slang-shaders/tree/master/crt/shaders/crt-royale)-Shaders (ursprünglich Slang/GLSL von TroggleMonkey, GPL v2+) nach Apple Metal Shading Language (MSL), integriert in die macOS-App [RetroVisor](https://github.com/dirkwhoffmann/RetroVisor) von Prof. Dr. Dirk W. Hoffmann.

**Master-Semesterprojekt, HKA.**<br>
Devin Uyan - 1. Semester Master Informatik.

## Stand der Pipeline

12 von 12 Slang-Passes portiert. <br>11 von 12 sind ΔE-validiert gegen die Referenz (`librashader-cli` mit der originalen Slang-Pipeline). <br>Die Default-Pipeline läuft mit ~1050 FPS auf Apple M2 bei 256×768.

| Slang-Pass | Beschreibung | Status |
|------------|--------------|--------|
| 0 | Linearize CRT-Gamma + Bob-Fields | Portiert + validiert |
| 1 | Vertikale Scanlines (Beam-Distribution) | Portiert + validiert |
| 2 | Bloom Approx (4×4 Gauss) | Portiert + validiert |
| 3+4 | Halation V/H (9-tap Gauss) | Portiert + validiert |
| 5+6 | Mask Resize V/H (Lanczos) | Portiert; Mode-0-Pfad nicht bit-exakt <br>(Default ist Mode 1) |
| 7 | Apply Mask + Halation | Portiert + validiert |
| 8 | Brightpass (Area-Bloom-Extract) | Portiert + validiert |
| 9+10 | Bloom V/H + Reconstitute | Portiert + validiert |
| 11 | Geometry + AA + Final Encode | Portiert <br>(Sphere-Raycaster + 16-tap Catmull-Rom-AA) |

## Verzeichnislayout

Dieses Repository (`crt-royale-msl`) liegt erwartungsgemäß **neben** zwei externen Repositories, <br>die getrennt geklont werden:

```
some-workspace/
├── crt-royale-msl/         ← dieses Repository
│   ├── integration/        # Source-of-Truth für die RetroVisor-Integration
│   │   ├── CrtRoyale.metal       # die produktive MSL-Pipeline (~1560 Zeilen)
│   │   ├── CrtRoyale.swift       # Swift-Integration (Settings, Kernel, Pipeline)
│   │   ├── textures/             # Phosphor-Mask-LUTs
│   │   ├── build-patches/        # Patches für die im Vendor-Repo getrackten Dateien
│   │   ├── setup.py              # Plattformübergreifendes Setup (macOS/Linux/Windows)
│   │   ├── setup.sh              # Bash-Wrapper
│   │   ├── setup.ps1             # PowerShell-Wrapper für Windows
│   │   └── README.md             # Detaillierte Erklärung des Setup-Mechanismus
│   ├── src/                # Studienmirror-Dateien pro Pass (Doku-Mirror, nicht gebaut)
│   ├── tests/
│   │   ├── inputs/               # Prozedural erzeugte Test-Patterns (Color Bars, Grid, ...)
│   │   ├── outputs/              # SwiftRunner-Output-Snapshots
│   │   ├── reference/            # librashader-cli-Referenz-Snapshots
│   │   ├── diff/                 # ΔE2000-Heatmaps + Statistiken
│   │   ├── SwiftRunner/          # Headless-Validierungs-Harness (Swift-Package)
│   │   ├── validate.sh           # Haupteinstiegspunkt für Tests
│   │   ├── capture_reference.sh  # Referenz-Snapshots neu erzeugen
│   │   └── compare_all.sh        # Batch-Vergleich Outputs vs. Referenz
│   ├── tools/
│   │   ├── compare.py            # ΔE2000 + SSIM + Heatmap
│   │   └── analyze.py            # Sanity-Checks für SwiftRunner-Outputs
│   ├── textures/                 # Arbeitskopien der Mask-LUTs
│   └── docs/                     # NICHT versioniert – privat (Bericht, Status etc.)
├── vendor/
│   ├── RetroVisor/         ← `git clone git@github.com:dirkwhoffmann/RetroVisor.git`
│   └── slang-shaders/      ← `git clone git@github.com:libretro/slang-shaders.git`
```

## Erste Schritte

### Voraussetzungen

- macOS 13+ (getestet auf Apple Silicon; Intel sollte funktionieren), <br>oder Linux für die Tests (App-Build nur unter macOS).
- Xcode 15+ (für `swift build` und MSL-Compilation auf macOS).
- Python 3.9+ mit `numpy`, `Pillow`, `scikit-image` (für `compare.py` und `analyze.py`).
- Rust + librashader-cli (optional, für Referenz-Snapshots).

### 1) Vendor-Repositories klonen (parallel zu diesem Repo)

```bash
cd ..   # raus aus crt-royale-msl/
mkdir -p vendor && cd vendor
git clone git@github.com:dirkwhoffmann/RetroVisor.git
git clone git@github.com:libretro/slang-shaders.git
cd ..
```

### 2) Integrations-Setup ausführen

Verlinkt unsere `CrtRoyale.metal` / `CrtRoyale.swift` / LUT als Symlinks in den `vendor/RetroVisor`-Pfad und wendet kleine Patches auf Xcode-Projekt, Storyboard und ShaderLibrary an. <br>Details siehe [`integration/README.md`](integration/README.md).

```bash
# macOS / Linux:
bash crt-royale-msl/integration/setup.sh

# Windows (PowerShell, Python im PATH):
pwsh crt-royale-msl/integration/setup.ps1
```

Das Setup ist idempotent und mit `--undo` voll rückgängig machbar.
<br>Auf Windows ohne Developer-Mode fällt es automatisch auf File-Copies statt Symlinks zurück <br>(oder explizit mit `--copy`).

### 3) RetroVisor-App bauen

```bash
cd vendor/RetroVisor
open RetroVisor.xcodeproj
# ⌘B, dann ⌘R
```

In der laufenden App: Shader-Picker → **CRT-Royale**.

## Headless-Validierung

```bash
bash crt-royale-msl/tests/validate.sh
```

Was passiert:
1. Deterministische Test-Inputs werden generiert (Color Bars, Gradient, Grid, ...).
2. Der Swift-Runner wird gebaut.
3. Die gesamte Pipeline läuft für jedes Input-PNG durch (12 Slang-Passes + Geometry-AA).
4. Der Output wird gegen die librashader-Referenz-Snapshots in `tests/reference/` verglichen.

<br>Erwartetes Resultat: **75 / 75 Sanity-Checks grün**; <br>ΔE-Tabelle gegen Slang-Referenz mit `04-final` Mean ΔE ≤ 5.04.

### Referenz-Snapshots neu erzeugen (selten nötig)

Wenn sich an der Slang-Quelle etwas ändert oder librashader-cli aktualisiert wird, <br>kann die Referenz neu generiert werden:

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

Die Validierung läuft in zwei Stufen:

1. **Sanity-Checks** (`tools/analyze.py`) - strukturelle Checks am SwiftRunner-Output: <br>Round-Trip-Identität von Pass 1 bei `--neutral`, Korrektheit des Y-Upscalings, <br>Wertebereichs-Prüfungen, Konsistenz der RGB-Triade auf Solid-Inputs.

2. **ΔE2000 + SSIM gegen Slang-Referenz** (`tools/compare.py`) - <br>pixelgenauer Vergleich gegen librashader-cli-Output. <br>Akzeptanzkriterium für portierte Stages: `mean ΔE2000 < 2.0`.

<br>Beide Werkzeuge werden in `validate.sh` orchestriert. Die SwiftRunner-Pipeline produziert pro Test-Input 13 Snapshots (`00-input` … `04-final`, dazu alle Zwischenstufen `02b-bloom_approx`, `02c-halation_v`, `02d-halation_blur`, `02e-mask_resize_v`, `02f-mask_resize`, `03-pass3`, `03b-brightpass`, `03c-bloom_v`, `03d-bloom_final`), so dass jeder Slang-Pass isoliert vergleichbar ist.

## Bekannte Einschränkungen

- **Mask-Resize Mode 0** (Lanczos-resized Mask-LUT) ist implementiert, aber die `mask_resize_tile_size`-Konstantenkette in librashader weicht von meiner Hand-Trace ab - <br>die Default-Pipeline läuft im Mode-1-Pfad (Hardware-Resample, ΔE-validiert).

- **Curvature-Branch** in Pass 11 ist nicht ΔE-validiert (benötigt eigene Referenz-Snapshots).
- **Geometrie-Modi 2 und 3** (Sphere_Alt, Cylinder) sind nicht portiert. <br>Sphere (Mode 1) deckt den üblichen Curved-CRT-Fall ab.
- **AA-Filter-Konfigurierbarkeit**: Catmull-Rom-Cubic ist fixiert <br>(Slang exponiert eine 10×11-Matrix aus Filtertypen und Sample-Counts). <br>Visuell äquivalent für die Moire-Suppression.

## Lizenzen

- Original CRT-Royale: GPL v2+ © 2014 TroggleMonkey.
- Diese Portierung: GPL v2+ (abgeleitetes Werk).
- RetroVisor: GPL v3 © Dirk W. Hoffmann.
