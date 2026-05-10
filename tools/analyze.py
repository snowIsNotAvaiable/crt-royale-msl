#!/usr/bin/env python3
"""Analyze runner outputs for the CRT-Royale MSL port.

Walks an outputs/ directory of subdirs (one per test pattern), each containing:
    00-input.png, 01-pass1.png, 02-pass2.png, 03-final.png
and runs sanity checks per pattern. Prints a PASS/FAIL report and exits non-
zero if any check fails.

Modes:
  --mode neutral  pass1 should round-trip to identity (crt_gamma == lcd_gamma)
  --mode default  pass1 only needs to preserve monotonicity (gamma simulated)

Pass 2 is at Y-upscaled resolution (4x source by default). The analyzer
uses this to verify that visible scanline structure exists in the output.

Requires: Pillow, numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print(f"analyze.py: missing dependency: {exc}", file=sys.stderr)
    print("Run:  pip3 install --user Pillow numpy", file=sys.stderr)
    sys.exit(2)


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def load(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        suffix = f" {DIM}-- {detail}{RESET}" if detail else ""
        print(f"    [{tag}] {name}{suffix}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1


def y_high_freq_energy(img: np.ndarray) -> float:
    """Mean |row_i - row_{i-1}| over all channels."""
    f = img.astype(np.int32)
    return float(np.abs(f[1:] - f[:-1]).mean())


def x_high_freq_energy(img: np.ndarray) -> float:
    f = img.astype(np.int32)
    return float(np.abs(f[:, 1:] - f[:, :-1]).mean())


def downsample_y(img: np.ndarray, factor: int) -> np.ndarray:
    h = (img.shape[0] // factor) * factor
    return img[:h].reshape(h // factor, factor, img.shape[1], -1).mean(axis=1).astype(np.uint8)


def analyze_pattern(name: str, dir: Path, rep: Report, mode: str) -> None:
    print(f"  {CYAN}{name}{RESET}  ({dir})")

    files = {
        "input": dir / "00-input.png",
        "pass1": dir / "01-pass1.png",
        "pass2": dir / "02-pass2.png",
        "pass3": dir / "03-pass3.png",
        "final": dir / "04-final.png",
    }
    for k, p in files.items():
        if not p.exists():
            rep.check(f"{k} present", False, f"missing {p.name}")
            return

    inp   = load(files["input"])
    pass1 = load(files["pass1"])
    pass2 = load(files["pass2"])
    pass3 = load(files["pass3"])
    final = load(files["final"])

    h_in, w_in = inp.shape[:2]
    h_p2, w_p2 = pass2.shape[:2]

    # Pass 1 stays at source res; Pass 2/final are Y-upscaled.
    rep.check("pass1 at source res",
              pass1.shape == inp.shape,
              f"input {inp.shape} vs pass1 {pass1.shape}")
    rep.check("pass2 same shape as final",
              pass2.shape == final.shape,
              f"pass2 {pass2.shape} vs final {final.shape}")
    rep.check("pass2 width matches input",
              w_p2 == w_in,
              f"input W={w_in} vs pass2 W={w_p2}")
    rep.check("pass2 Y-upscaled vs input",
              h_p2 >= h_in,
              f"input H={h_in} vs pass2 H={h_p2}")

    rep.check("pass1 values in [0,255]",
              0 <= int(pass1.min()) and int(pass1.max()) <= 255,
              f"min={int(pass1.min())} max={int(pass1.max())}")
    rep.check("pass2 values in [0,255]",
              0 <= int(pass2.min()) and int(pass2.max()) <= 255,
              f"min={int(pass2.min())} max={int(pass2.max())}")
    rep.check("pass3 values in [0,255]",
              0 <= int(pass3.min()) and int(pass3.max()) <= 255,
              f"min={int(pass3.min())} max={int(pass3.max())}")

    # Pass 1 round-trip vs input.
    if mode == "neutral":
        diff = np.abs(pass1.astype(np.int32) - inp.astype(np.int32))
        mean_err = float(diff.mean())
        max_err = int(diff.max())
        rep.check("pass1 ≈ input (round-trip identity)",
                  mean_err < 2.0 and max_err < 8,
                  f"mean={mean_err:.2f} max={max_err}")
    else:
        # Default mode: not identity, but pass1 must preserve ordering.
        gray_in = inp.astype(np.float32).mean(axis=2)
        gray_p1 = pass1.astype(np.float32).mean(axis=2)
        rng = np.random.default_rng(seed=42)
        n_total = gray_in.size
        idx_a = rng.integers(0, n_total, size=512)
        idx_b = rng.integers(0, n_total, size=512)
        ok_pairs = 0
        n_pairs = 0
        flat_in = gray_in.flatten()
        flat_p1 = gray_p1.flatten()
        for a, b in zip(idx_a, idx_b):
            if abs(flat_in[a] - flat_in[b]) < 4:
                continue
            n_pairs += 1
            if (flat_in[a] >= flat_in[b]) == (flat_p1[a] >= flat_p1[b]):
                ok_pairs += 1
        if n_pairs == 0:
            # Solid input: nothing to order, trivially preserved.
            rep.check("pass1 preserves ordering",
                      True, "solid input, trivially preserved")
        else:
            ratio = ok_pairs / n_pairs
            rep.check("pass1 preserves ordering",
                      ratio > 0.95,
                      f"{ok_pairs}/{n_pairs} pairs preserved ({ratio:.1%})")

    # --- Pass 2 / scanline checks ---

    is_solid = name in {"solid_white", "solid_black", "solid_gray"}
    is_horiz_struct = name == "horiz_lines"

    if is_solid:
        # On a solid input, pass2 must be horizontally uniform (low X-energy)
        # and produce zero-amplitude scanlines on solid_black, autodim-only on
        # solid_white, etc.
        x_e = x_high_freq_energy(pass2)
        rep.check("pass2 X-uniform on solid input",
                  x_e < 1.0,
                  f"X-energy={x_e:.3f}")
        if name == "solid_black":
            rep.check("solid_black stays dark",
                      int(pass2.max()) < 8, f"max={int(pass2.max())}")
        if name == "solid_white":
            m = float(pass2.mean())
            rep.check("solid_white not fully clipped",
                      20 < m < 240, f"mean={m:.1f}")
        if name == "solid_gray":
            m = float(pass2.mean())
            rep.check("solid_gray within plausible band",
                      5 < m < 200, f"mean={m:.1f}")
    else:
        # For non-solid inputs without strong existing Y-structure, pass 2
        # must add scanline modulation. Inputs that already have strong
        # vertical structure (grids, horizontal-line patterns) hide pass 2's
        # contribution, so we only run this check when the source itself is
        # nearly Y-uniform (gradient, colorbars, ...).
        y_e_in_source = y_high_freq_energy(inp)
        if y_e_in_source < 5.0:
            y_e_p2 = y_high_freq_energy(pass2)
            rep.check("pass2 adds scanlines on Y-uniform source",
                      y_e_p2 > 5.0,
                      f"input Y-energy={y_e_in_source:.1f} -> pass2={y_e_p2:.1f}")
        else:
            rep.check("pass2 adds scanlines on Y-uniform source",
                      True,
                      f"skipped: source has existing Y-structure (Y-energy={y_e_in_source:.1f})")

    # --- Pass 3 / mask checks ---
    # Pass 3 multiplies pass-2 by a procedural aperture grille (3-px triad,
    # one R/G/B subpixel per pixel column). Expectation: pass 3 introduces
    # strong X-direction structure (per-channel periodicity) that wasn't
    # there in pass 2.
    if name == "solid_black":
        rep.check("pass3 stays dark on solid_black",
                  int(pass3.max()) < 8, f"max={int(pass3.max())}")
    else:
        # The aperture-grille mask quintessentially adds X-direction subpixel
        # structure. Skip the energy-delta test when the input already had
        # heavy X-variation (color bars, anything channel-checkered) -- the
        # mask multiplies into that variation rather than dominating it.
        # The "RGB triad pattern is consistent" check below is the stronger
        # invariant and runs unconditionally on near-solid inputs.
        x_e_p2 = x_high_freq_energy(pass2)
        x_e_p3 = x_high_freq_energy(pass3)
        if x_e_p2 < 10.0:
            rep.check("pass3 introduces subpixel X-structure",
                      x_e_p3 > x_e_p2 + 5.0,
                      f"X-energy: pass2={x_e_p2:.1f} -> pass3={x_e_p3:.1f}")
        else:
            rep.check("pass3 introduces subpixel X-structure",
                      True,
                      f"skipped: pass2 already has X-structure ({x_e_p2:.1f})")

        # Triad period: with default mask_triad_size=3, expect peak structure
        # at exactly 3-pixel period in X. Each color channel should peak in
        # one specific column-mod-3 bucket. Use solid-ish inputs to validate.
        if name in {"solid_white", "solid_gray"}:
            # Pick the brightest row (avoid scanline gaps), then check that
            # column index mod 3 controls which channel dominates.
            row_brightness = pass3.astype(np.float32).sum(axis=2).mean(axis=1)
            best_y = int(np.argmax(row_brightness))
            row = pass3[best_y]  # shape (W, 3)
            ch_dominant = np.argmax(row, axis=1)  # 0=R, 1=G, 2=B per column
            # In each col-mod-3 bucket, check the dominant channel is consistent.
            consistent = True
            for bucket in range(3):
                vals = ch_dominant[bucket::3]
                # Most common channel in this bucket should hold > 80%.
                counts = np.bincount(vals, minlength=3)
                if counts.max() / counts.sum() < 0.8:
                    consistent = False
                    break
            rep.check("pass3 RGB triad pattern is consistent",
                      consistent,
                      f"row {best_y}: bucket dominance >= 80%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs_dir")
    ap.add_argument("--mode", choices=("neutral", "default"), default="neutral")
    args = ap.parse_args()

    root = Path(args.outputs_dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    rep = Report()
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        print(f"no pattern subdirs under {root}", file=sys.stderr)
        return 2

    print(f"\n{CYAN}=== CRT-Royale validation (mode={args.mode}) ==={RESET}\n")
    for d in subdirs:
        analyze_pattern(d.name, d, rep, args.mode)
        print()

    total = rep.passed + rep.failed
    color = GREEN if rep.failed == 0 else RED
    print(f"{color}{rep.passed}/{total} checks passed, {rep.failed} failed{RESET}")
    return 0 if rep.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
