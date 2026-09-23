#!/usr/bin/env python3
"""Widen laya's f16 checkpoint to f32 for the ndarray CPU backend.

The GPU (wgpu) path loads the ORIGINAL f16 safetensors directly: burn-store's
DTypePolicy::FromRecord adopts each tensor's dtype verbatim, and the Rust model
transposes torch-layout `[out, in]` Linear weights at use, so no layout or dtype
preparation is needed there. This script only widens dtype (f16 -> f32 is
exact); keys and layouts are untouched. The f16 path never needs to run this.

Usage: python3 scripts/convert_checkpoint.py <in.safetensors> <out.safetensors>
"""
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    tensors = load_file(src)
    wide = {k: v.astype(np.float32) for k, v in tensors.items()}
    save_file(wide, dst)
    print(f"wrote {dst} ({len(wide)} tensors, F32)")


if __name__ == "__main__":
    main()