#!/usr/bin/env python3
"""Erzeugt publikationsreife Abbildungen für den LaTeX-Bericht aus den
Validierungs-Artefakten (tests/diff/*.json, tests/outputs, tests/reference).

Output: docs/Bericht/figures/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIFF = ROOT / "tests" / "diff" / "default"
OUT = ROOT / "docs" / "Bericht" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Design-Parameter (dataviz reference palette, light mode) ---------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
})

PATTERNS = ["colorbars", "gradient", "grid", "horiz_lines",
            "solid_black", "solid_gray", "solid_white"]
STAGES = ["02-pass2", "02b-bloom_approx", "02c-halation_v", "02d-halation_blur",
          "03-pass3", "03b-brightpass", "03c-bloom_v", "03d-bloom_final",
          "04-final"]
STAGE_LABELS = ["Scanlines\n(P1)", "Bloom-\nApprox (P2)", "Halation V\n(P3)",
                "Halation H\n(P4)", "Apply Mask\n(P7)", "Brightpass\n(P8)",
                "Bloom V\n(P9)", "Bloom Final\n(P10)", "Final\n(P11)"]


def load(pattern: str, stage: str):
    p = DIFF / pattern / f"{stage}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def strip_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Figur 1 --
# Mean-ΔE2000 der Gesamt-Pipeline (04-final) pro Test-Pattern, mit SSIM-Label
def fig_deltae_final():
    data = [(p, load(p, "04-final")) for p in PATTERNS]
    names = [p for p, d in data]
    means = [d["delta_e_2000"]["mean"] for _, d in data]
    ssims = [d["ssim"] for _, d in data]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(names))[::-1]
    colors = [BLUE if m < 2.0 else "#6da7ec" for m in means]
    bars = ax.barh(y, means, height=0.62, color=colors, zorder=3)
    for yi, m, s in zip(y, means, ssims):
        ax.text(m + 0.07, yi, f"{m:.2f}", va="center", ha="left",
                fontsize=9, color=INK)
        ax.text(5.55, yi, f"SSIM {s:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK2)
    ax.axvline(2.0, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(2.0, len(names) - 0.28, " Akzeptanzkriterium $\\bar{\\Delta E} < 2{,}0$",
            color=RED, fontsize=8.5, va="bottom", ha="left")
    ax.set_yticks(y, names, fontsize=9.5, color=INK)
    ax.set_xlim(0, 6.4)
    ax.set_xlabel("mittleres $\\Delta E_{2000}$ (MSL-Port vs. Slang-Referenz)")
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    strip_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig-deltae-final.pdf")
    fig.savefig(OUT / "fig-deltae-final.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figur 2 --
# Heatmap: mean ΔE2000 pro (Pattern × Stage)
def fig_stage_matrix():
    M = np.full((len(PATTERNS), len(STAGES)), np.nan)
    for i, p in enumerate(PATTERNS):
        for j, s in enumerate(STAGES):
            d = load(p, s)
            if d:
                M[i, j] = d["delta_e_2000"]["mean"]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    clipped = np.clip(M, 0, 8)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(clipped, cmap=cmap, vmin=0, vmax=8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if clipped[i, j] > 4.4 else INK)
    ax.set_xticks(range(len(STAGES)), STAGE_LABELS, fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(PATTERNS)), PATTERNS, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # weiße Trennfugen
    ax.set_xticks(np.arange(-0.5, len(STAGES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PATTERNS)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label("mittleres $\\Delta E_{2000}$ (≥ 8 gesättigt)", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig-deltae-stages.pdf")
    fig.savefig(OUT / "fig-deltae-stages.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figur 3 --
# Performance-Benchmark (p50-GPU-Zeit) mit 60-FPS-Budget als Referenz
def fig_performance():
    configs = ["Default\n(256×768)", "8× Y-Upscale\n(256×1536)",
               "Curvature + AA\n(256×768)", "Curvature + AA, 8×\n(256×1536)"]
    p50 = [0.95, 1.73, 1.18, 1.13]
    fps = [1049, 579, 848, 886]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(configs))
    ax.bar(x, p50, width=0.56, color=BLUE, zorder=3)
    for xi, v, f in zip(x, p50, fps):
        ax.text(xi, v + 0.05, f"{v:.2f} ms", ha="center", fontsize=9, color=INK)
        ax.text(xi, v / 2, f"≈{f}\nFPS", ha="center", va="center",
                fontsize=8.5, color="#ffffff", linespacing=1.1)
    ax.axhline(16.67, color=AQUA, lw=1.2, ls=(0, (4, 3)))
    ax.text(len(configs) - 0.52, 16.67 * 0.86, "60-FPS-Budget (16,67 ms)",
            color="#0e7a55", fontsize=8.5, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 24)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16], ["0,5", "1", "2", "4", "8", "16"])
    ax.set_ylabel("GPU-Zeit p50 [ms], log-Skala")
    ax.set_xticks(x, configs, fontsize=8.5, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    strip_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig-performance.pdf")
    fig.savefig(OUT / "fig-performance.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figur 4 --
# Side-by-side: MSL-Output vs. Slang-Referenz vs. ΔE-Heatmap (2 Patterns)
def fig_side_by_side():
    rows = ["colorbars", "grid"]
    cols = [
        ("MSL-Port (Metal)", lambda p: ROOT / "tests/outputs/default" / p / "04-final.png"),
        ("Slang-Referenz (librashader)", lambda p: ROOT / "tests/reference" / p / "04-final.png"),
        ("$\\Delta E_{2000}$-Heatmap", lambda p: DIFF / p / "04-final-heat.png"),
    ]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.8, 6.4))
    for i, pat in enumerate(rows):
        for j, (title, fn) in enumerate(cols):
            ax = axes[i, j]
            img = Image.open(fn(pat))
            ax.imshow(np.asarray(img), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if i == 0:
                ax.set_title(title, fontsize=9.5, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(pat, fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig-final-comparison.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figur 5 --
# Scanline/Mask-Detail: Zoom in den finalen Output (Phosphor-Triade sichtbar)
def fig_detail_zoom():
    src = Image.open(ROOT / "tests/outputs/default/colorbars/04-final.png")
    w, h = src.size
    crop = src.crop((0, h // 3, 96, h // 3 + 72))          # 96×72-Ausschnitt
    zoom = crop.resize((crop.width * 8, crop.height * 8), Image.NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1),
                             gridspec_kw={"width_ratios": [1, 2.2]})
    axes[0].imshow(np.asarray(src), interpolation="nearest")
    axes[0].add_patch(plt.Rectangle((0, h // 3), 96, 72, fill=False,
                                    edgecolor=RED, lw=1.4))
    axes[0].set_title("Finaler Output (256×768)", fontsize=9.5, pad=6)
    axes[1].imshow(np.asarray(zoom), interpolation="nearest")
    axes[1].set_title("8×-Zoom: Scanlines + Phosphor-Triade", fontsize=9.5, pad=6)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig-detail-zoom.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_deltae_final()
    fig_stage_matrix()
    fig_performance()
    fig_side_by_side()
    fig_detail_zoom()
    print("Figuren geschrieben nach", OUT)
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)
