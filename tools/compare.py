#!/usr/bin/env python3
"""compare.py -- Per-Pixel-Differenz zwischen zwei Snapshots.

Vergleicht zwei PNGs (z.B. unsere MSL-Pipeline-Ausgabe vs. RetroArch-Reference)
mit perzeptuellen Metriken:
  - Delta E 2000 (CIE 2000 color difference, perceptually uniform)
  - SSIM (Structural Similarity Index)
  - RGB MSE / max diff (numerischer Sanity-Check)

Optional schreibt es eine Heatmap-PNG, die Delta E pro Pixel visualisiert
(schwarz = identisch, hell-rot = stark abweichend). JSON-Stats auf Wunsch.

Usage:
    compare.py --left A.png --right B.png \\
               [--heatmap diff.png] [--json stats.json] \\
               [--threshold 2.0]

Exit codes:
    0  identisch oder unter --threshold (delta_e_2000.mean)
    1  Differenzen ueber threshold
    2  IO / shape mismatch

Requires: numpy, Pillow, colour-science, scikit-image
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _silence_stderr():
    """Briefly redirect stderr to /dev/null around noisy imports."""
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


try:
    import numpy as np
    from PIL import Image
    # colour-science transitively imports matplotlib, which in our env
    # triggers a numpy 1.x/2.x ABI mismatch warning on stderr that we
    # don't care about (we never call into colour.plotting).
    with _silence_stderr():
        import colour
    from skimage.metrics import structural_similarity as ssim
except ImportError as exc:
    print(f"compare.py: missing dependency: {exc}", file=sys.stderr)
    print("Run:  pip3 install --user numpy Pillow colour-science scikit-image",
          file=sys.stderr)
    sys.exit(2)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def srgb_uint8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 [0..255] to CIELAB. Vectorised."""
    rgb = rgb_u8.astype(np.float64) / 255.0
    xyz = colour.sRGB_to_XYZ(rgb)
    return colour.XYZ_to_Lab(xyz)


def render_heatmap_png(de_per_pixel: np.ndarray, path: Path,
                       de_max: float = 10.0) -> None:
    """Black -> red linear mapping; values >= de_max saturate red.

    The 0..de_max range covers "imperceptible" through "clearly visible"
    differences -- standard interpretation of Delta E in JNDs."""
    norm = np.clip(de_per_pixel / max(de_max, 1e-6), 0.0, 1.0)
    img = np.zeros((*de_per_pixel.shape, 3), dtype=np.uint8)
    img[..., 0] = (norm * 255.0).astype(np.uint8)  # red channel
    # Add a low-light green tint for very small differences so 0 vs near-0
    # isn't entirely indistinguishable.
    img[..., 1] = ((norm > 0) * np.minimum(norm * 64.0, 32.0)).astype(np.uint8)
    Image.fromarray(img, "RGB").save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, type=Path,
                    help="path to one PNG (typically our MSL output)")
    ap.add_argument("--right", required=True, type=Path,
                    help="path to the other PNG (typically the reference)")
    ap.add_argument("--heatmap", type=Path, default=None,
                    help="optional Delta E heatmap output PNG")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional JSON stats output")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="exit non-zero if mean Delta E exceeds this "
                         "(default: 2.0 = 'just perceptible')")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress non-error console output")
    args = ap.parse_args()

    if not args.left.is_file():
        print(f"missing: {args.left}", file=sys.stderr); return 2
    if not args.right.is_file():
        print(f"missing: {args.right}", file=sys.stderr); return 2

    a = load_rgb(args.left)
    b = load_rgb(args.right)
    if a.shape != b.shape:
        print(f"shape mismatch: {args.left.name}={a.shape} vs "
              f"{args.right.name}={b.shape}", file=sys.stderr)
        return 2

    a_lab = srgb_uint8_to_lab(a)
    b_lab = srgb_uint8_to_lab(b)
    de = colour.delta_E(a_lab, b_lab, method="CIE 2000")

    s = float(ssim(a, b, channel_axis=2, data_range=255))

    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    rgb_mse = float(np.sqrt((diff.astype(np.float64) ** 2).mean()))
    rgb_max = int(diff.max())

    stats = {
        "left":  str(args.left),
        "right": str(args.right),
        "shape": list(a.shape),
        "delta_e_2000": {
            "mean": float(de.mean()),
            "p50":  float(np.percentile(de, 50)),
            "p95":  float(np.percentile(de, 95)),
            "p99":  float(np.percentile(de, 99)),
            "max":  float(de.max()),
        },
        "ssim": s,
        "rgb_mse": rgb_mse,
        "rgb_max_diff": rgb_max,
    }

    if not args.quiet:
        print(json.dumps(stats, indent=2))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2))

    if args.heatmap:
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        render_heatmap_png(de, args.heatmap)
        if not args.quiet:
            print(f"heatmap: {args.heatmap}")

    # Conventional reading of Delta E:
    #   < 1   imperceptible
    #   1..2  perceptible on close inspection
    #   2..10 perceptible at a glance
    #   > 10  clearly different
    return 0 if stats["delta_e_2000"]["mean"] <= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
