#!/usr/bin/env python3
"""Generate deterministic test inputs for the CRT-Royale validation pipeline.

Each pattern stresses a different shader behavior:
    colorbars.png  -- gamma & per-channel correctness
    grid.png       -- scanline alignment, geometry stability
    gradient.png   -- gamma curve smoothness, banding
    solid_white    -- max-brightness clipping, autodim correctness
    solid_black    -- darks must stay dark (no NaN/sign flip)
    solid_gray     -- mid-luminance round-trip sanity
    horiz_lines    -- per-line response (Pass 2 should keep these intact)

Outputs are 256x192 PNGs (NES-like resolution) written next to this script.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path

W = 256
H = 192


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(
        ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
    )


def write_png(path: Path, pixels: list[tuple[int, int, int]]) -> None:
    """Write an 8-bit RGB PNG. Avoids needing PIL."""
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type 0 (None)
        for x in range(W):
            r, g, b = pixels[y * W + x]
            raw.extend((r, g, b))
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    body = _png_chunk(b"IHDR", ihdr)
    body += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    body += _png_chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)


def colorbars() -> list[tuple[int, int, int]]:
    bars = [
        (255, 255, 255),  # white
        (255, 255, 0),    # yellow
        (0, 255, 255),    # cyan
        (0, 255, 0),      # green
        (255, 0, 255),    # magenta
        (255, 0, 0),      # red
        (0, 0, 255),      # blue
        (0, 0, 0),        # black
    ]
    pix = []
    bar_w = W // len(bars)
    for y in range(H):
        for x in range(W):
            pix.append(bars[min(x // bar_w, len(bars) - 1)])
    return pix


def grid() -> list[tuple[int, int, int]]:
    pix = []
    for y in range(H):
        for x in range(W):
            on_line = (x % 16 == 0) or (y % 16 == 0)
            pix.append((255, 255, 255) if on_line else (0, 0, 0))
    return pix


def gradient() -> list[tuple[int, int, int]]:
    pix = []
    for y in range(H):
        for x in range(W):
            v = int(255 * x / (W - 1))
            pix.append((v, v, v))
    return pix


def solid(color: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [color] * (W * H)


def horiz_lines() -> list[tuple[int, int, int]]:
    pix = []
    for y in range(H):
        on = (y % 4) < 2
        c = (255, 255, 255) if on else (0, 0, 0)
        pix.extend(c for _ in range(W))
    return pix


PATTERNS = {
    "colorbars": colorbars,
    "grid": grid,
    "gradient": gradient,
    "solid_white": lambda: solid((255, 255, 255)),
    "solid_black": lambda: solid((0, 0, 0)),
    "solid_gray":  lambda: solid((128, 128, 128)),
    "horiz_lines": horiz_lines,
}


def main() -> int:
    out_dir = Path(__file__).parent
    for name, fn in PATTERNS.items():
        target = out_dir / f"{name}.png"
        write_png(target, fn())
        print(f"  wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
