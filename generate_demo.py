"""Generate a self-contained synthetic demo for the stereo-mosaicing pipeline.

Builds a wide, richly textured scene, pans a viewport horizontally across it to
fake a moving-camera clip (pure X translation with mild handheld Y jitter), then
runs the manifold-mosaicing core in ``mosaic.py`` to produce the wiggle-stereo
panorama. Writes everything to ``videos/`` so the repo runs end-to-end without
any external footage.

    python generate_demo.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import mediapy
import numpy as np

from mosaic import generate_panorama

RNG = np.random.default_rng(7)
VIEW_W, VIEW_H = 640, 400
N_FRAMES = 80
N_OUT = 30
STEP = 18                  # background pan per frame (px)
FG_PARALLAX = 1.7          # foreground pans faster -> depth parallax -> wiggle stereo


def _scatter(layer, mask, n, kinds, size_range, rng):
    """Draw n random shapes onto an RGB layer + its alpha mask."""
    h, w = layer.shape[:2]
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(c) for c in rng.integers(40, 256, 3))
        lo, hi = size_range
        if rng.integers(0, 2) == 0:
            r = int(rng.integers(lo, hi))
            cv2.circle(layer, (x, y), r, color, -1)
            cv2.circle(mask, (x, y), r, 255, -1)
        else:
            wd, ht = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
            cv2.rectangle(layer, (x, y), (x + wd, y + ht), color, -1)
            cv2.rectangle(mask, (x, y), (x + wd, y + ht), 255, -1)


def build_layers(pan: int):
    """Build a far background and a near foreground, each wide enough to pan."""
    bg_w = VIEW_W + pan + 8
    fg_w = VIEW_W + int(pan * FG_PARALLAX) + 8
    # far background: sky->ground gradient + lots of small distant detail
    col = np.linspace(210, 70, VIEW_H).astype(np.uint8)
    bg = np.repeat(col[:, None, None], bg_w, axis=1).repeat(3, axis=2).copy()
    bg[:, :, 2] = np.clip(bg[:, :, 2].astype(int) + 25, 0, 255)  # bluish sky tint
    _scatter(bg, np.zeros((VIEW_H, bg_w), np.uint8), 700, None, (3, 12), RNG)
    bg = np.clip(bg.astype(np.int16) + RNG.integers(-6, 7, bg.shape), 0, 255).astype(np.uint8)
    # near foreground: a few big objects on a transparent layer
    fg = np.zeros((VIEW_H, fg_w, 3), np.uint8)
    fg_mask = np.zeros((VIEW_H, fg_w), np.uint8)
    _scatter(fg, fg_mask, 40, None, (45, 120), RNG)
    return bg, fg, fg_mask


def render_frames(out_dir: str) -> list[np.ndarray]:
    """Composite background + parallax foreground per frame; save frame_*.jpg."""
    pan = STEP * (N_FRAMES - 1)
    bg, fg, fg_mask = build_layers(pan)
    frames = []
    for i in range(N_FRAMES):
        bx = int(round(i * STEP))
        fx = int(round(i * STEP * FG_PARALLAX))
        frame = bg[:, bx:bx + VIEW_W].copy()
        fg_slice = fg[:, fx:fx + VIEW_W]
        m = fg_mask[:, fx:fx + VIEW_W].astype(bool)
        frame[m] = fg_slice[m]
        frame = np.clip(frame.astype(np.int16) + RNG.integers(-4, 5, frame.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(frame)
    return frames


def main() -> None:
    out = Path(__file__).parent / "videos"
    out.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        input_frames = render_frames(tmp)
        mediapy.write_video(out / "demo_input.mp4", input_frames, fps=20)
        print(f"wrote {out / 'demo_input.mp4'}  ({len(input_frames)} frames)")

        result = generate_panorama(tmp, n_out_frames=N_OUT)

    result_frames = [np.asarray(im) for im in result]
    mediapy.write_video(out / "demo_result.mp4", result_frames, fps=12)
    print(f"wrote {out / 'demo_result.mp4'}  ({len(result_frames)} panorama frames, {result_frames[0].shape[1]}x{result_frames[0].shape[0]})")

    cv2.imwrite(str(out / "panorama_still.png"), cv2.cvtColor(result_frames[N_OUT // 2], cv2.COLOR_RGB2BGR))
    print(f"wrote {out / 'panorama_still.png'}")


if __name__ == "__main__":
    main()
