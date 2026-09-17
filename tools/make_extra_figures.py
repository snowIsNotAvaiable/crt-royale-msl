#!/usr/bin/env python3
"""Erzeugt Zusatz-Abbildungen + GIFs für Bericht und Webpage:

  fig-mask-types.png     -- Zoom-Vergleich Grille / Slot / Shadow
  fig-curvature.png      -- Flat vs. Curvature+AA (Demo-Modus)
  fig-pipeline-strip.png -- Filmstreifen aller 13 Pipeline-Stages
  pipeline-stages.gif    -- animierte Stage-Abfolge (Webpage)
  mask-types.gif         -- animierter Mask-Typ-Wechsel (Webpage)

Aufruf:  python3 tools/make_extra_figures.py [SCRATCH_DIR]
SCRATCH_DIR muss out-slot/, out-shadow/, out-demo/ aus SwiftRunner-Läufen
enthalten (siehe README); ohne Argument werden nur die Figuren erzeugt,
die allein aus tests/outputs/ ableitbar sind.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Bericht" / "figures"
WEB = ROOT / "docs" / "webpage"
OUT.mkdir(parents=True, exist_ok=True)
WEB.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = ROOT / "tests" / "outputs" / "default" / "colorbars"

# Maskentyp-Varianten: erzeugt von tests/capture_mask_variants.sh, also im Repo
# reproduzierbar. Frueher kam das aus einem Scratch-Verzeichnis, das als
# Argument uebergeben wurde; damit war die Abbildung nicht nachbaubar.
MASK_OUT = ROOT / "tests" / "outputs"

SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else None

INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e1e0d9"; RED = "#e34948"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff", "font.size": 10,
})

STAGES = ["00-input", "01-pass1", "02-pass2", "02b-bloom_approx", "02c-halation_v",
          "02d-halation_blur", "02e-mask_resize_v", "02f-mask_resize", "03-pass3",
          "03b-brightpass", "03c-bloom_v", "03d-bloom_final", "04-final"]
STAGE_SHORT = ["Input", "P0 Lin.", "P1 Scan", "P2 Bloom~", "P3 Hal.V", "P4 Hal.H",
               "P5 MaskV", "P6 MaskH", "P7 Mask", "P8 Bright", "P9 BloomV",
               "P10 Bloom", "P11 Final"]


def zoom(img: Image.Image, box, factor=6):
    c = img.crop(box)
    return c.resize((c.width * factor, c.height * factor), Image.NEAREST)


def mask_variant_paths(shadow_label="Shadow Mask (EDP)"):
    variants = [
        ("Aperture Grille", MASK_OUT / "mask-grille" / "04-final.png"),
        ("Slot Mask",       MASK_OUT / "mask-slot"   / "04-final.png"),
        (shadow_label,      MASK_OUT / "mask-shadow" / "04-final.png"),
    ]
    if all(p.exists() for _, p in variants):
        return variants
    print("  (uebersprungen: tests/capture_mask_variants.sh zuerst laufen lassen)")
    return None


MASK_LUTS = [
    ROOT / "integration" / "textures"
         / "TileableLinearApertureGrille15Wide8And5d5Spacing.png",
    ROOT / "integration" / "textures"
         / "TileableLinearSlotMaskTall15Wide9And4d5Horizontal9d14VerticalSpacing.png",
    ROOT / "integration" / "textures" / "TileableLinearShadowMaskEDP.png",
]


def fig_mask_types():
    """Zwei Reihen: oben die LUT-Textur, unten der Pipeline-Output.

    Die untere Reihe allein war irrefuehrend: bei der Slang-Default-Triadengroesse
    von 3 Pixeln mittelt das Mipmap-Sampling die LUT-Struktur so weit weg, dass
    Grille und Slot im Output fast identisch sind (mittlere RGB-Differenz 0.8 von
    255). Die obere Reihe zeigt, dass die Masken sehr wohl unterschiedlich sind,
    und macht den Effekt damit erst interpretierbar.
    """
    variants = mask_variant_paths()
    if variants is None:
        return
    if not all(p.exists() for p in MASK_LUTS):
        print("  (uebersprungen: Mask-LUTs fehlen unter integration/textures/)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for col, ((name, out_png), lut_png) in enumerate(zip(variants, MASK_LUTS)):
        # Obere Reihe: ein Tile der LUT (512 / 8 Triaden = 64 px), 4x vergroessert.
        lut = Image.open(lut_png).convert("RGB")
        axes[0][col].imshow(np.asarray(zoom(lut, (0, 0, 64, 48), factor=4)),
                            interpolation="nearest")
        axes[0][col].set_title(name, fontsize=9.5, pad=6)

        # Untere Reihe: derselbe Bildausschnitt des fertigen Outputs.
        axes[1][col].imshow(np.asarray(zoom(Image.open(out_png), (8, 300, 72, 348))),
                            interpolation="nearest")

    axes[0][0].set_ylabel("LUT-Textur", fontsize=9)
    axes[1][0].set_ylabel("Pipeline-Output", fontsize=9)
    for row in axes:
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig-mask-types.png", dpi=220)
    plt.close(fig)


def fig_curvature():
    if SCRATCH is None:
        return
    flat = Image.open(DEFAULT_OUT / "04-final.png")
    demo = Image.open(SCRATCH / "out-demo" / "04-final.png")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 4.6))
    for ax, (name, img) in zip(axes, [("geom_mode = 0 (flat)", flat),
                                      ("Demo: Sphere-Raycast + AA + Border", demo)]):
        ax.imshow(np.asarray(img), interpolation="bilinear")
        ax.set_title(name, fontsize=9.5, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig-curvature.png", dpi=220)
    plt.close(fig)


def fig_pipeline_strip():
    """Alle 13 Stages als 2x7-Raster; jede Zelle letterboxed auf Einheitsgröße."""
    cell_w, cell_h = 220, 240
    fig, axes = plt.subplots(2, 7, figsize=(7.2, 3.3))
    for idx, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if idx >= len(STAGES):
            ax.axis("off")
            continue
        img = Image.open(DEFAULT_OUT / f"{STAGES[idx]}.png").convert("RGB")
        s = min(cell_w / img.width, cell_h / img.height)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.NEAREST)
        canvas = Image.new("RGB", (cell_w, cell_h), (252, 252, 251))
        canvas.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
        ax.imshow(np.asarray(canvas), interpolation="nearest")
        ax.set_title(STAGE_SHORT[idx], fontsize=7, pad=3, color=INK2)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig-pipeline-strip.png", dpi=260)
    plt.close(fig)


def gif_pipeline():
    """Animierte Stage-Abfolge (gleiche Leinwand, Stage-Name eingeblendet)."""
    W, H = 360, 300
    frames = []
    for st, short in zip(STAGES, STAGE_SHORT):
        img = Image.open(DEFAULT_OUT / f"{st}.png").convert("RGB")
        s = min((W - 20) / img.width, (H - 46) / img.height)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.NEAREST)
        canvas = Image.new("RGB", (W, H), (10, 12, 16))
        canvas.paste(img, ((W - img.width) // 2, 38 + (H - 46 - img.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.text((12, 10), f"{short}   ({st})", fill=(67, 214, 117))
        frames.append(canvas)
    frames[0].save(WEB / "pipeline-stages.gif", save_all=True,
                   append_images=frames[1:], duration=900, loop=0)


def gif_mask_types():
    variants = mask_variant_paths("Shadow Mask")
    if variants is None:
        return
    frames = []
    for name, p in variants:
        z = zoom(Image.open(p), (8, 300, 88, 360), factor=5).convert("RGB")
        canvas = Image.new("RGB", (z.width, z.height + 30), (10, 12, 16))
        canvas.paste(z, (0, 30))
        d = ImageDraw.Draw(canvas)
        d.text((10, 8), name, fill=(67, 214, 117))
        frames.append(canvas)
    frames[0].save(WEB / "mask-types.gif", save_all=True,
                   append_images=frames[1:], duration=1100, loop=0)


if __name__ == "__main__":
    fig_pipeline_strip()
    gif_pipeline()
    fig_mask_types()
    fig_curvature()
    gif_mask_types()
    print("Zusatz-Figuren geschrieben:")
    for d in (OUT, WEB):
        for f in sorted(d.iterdir()):
            print("  ", f.relative_to(ROOT))
