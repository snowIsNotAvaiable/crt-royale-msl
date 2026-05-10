#!/usr/bin/env python3
"""compare.py -- Pixel-Diff zwischen RetroVisor- und RetroArch-Snapshots.

Slot reserved for the per-pass validation harness. Inputs are pairs of PNGs
(one from RetroVisor's "Save Snapshot" feature, one from a RetroArch dump
of the same shader pass), produced for the same emulated frame. Outputs
will include:

  - Per-pixel ΔE2000 heatmap (PNG)
  - SSIM score
  - Channel-wise histograms
  - Max / mean error summary

Implementation TODO. The Pass-2 debug picker + snapshot export already
provide the capture surface; this script is the analysis layer.

Usage (planned):
    python compare.py --retrovisor pass2-20260510-143055.png \\
                       --retroarch  pass2-reference.png \\
                       --output     diff-pass2.png
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrovisor", required=True,
                        help="Path to RetroVisor snapshot PNG")
    parser.add_argument("--retroarch", required=True,
                        help="Path to RetroArch reference PNG")
    parser.add_argument("--output", required=True,
                        help="Path to write the diff heatmap PNG")
    parser.parse_args()

    print("compare.py: not yet implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
