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
SCENE_W, SCENE_H = 2400, 480
VIEW_W, VIEW_H = 640, 400
N_FRAMES = 80
N_OUT = 30


def build_scene() -> np.ndarray:
    """A wide textured backdrop with plenty of corners for optical flow."""
    grad = np.linspace(40, 130, SCENE_W).astype(np.uint8)
    scene = np.repeat(grad[None, :, None], SCENE_H, axis=0).repeat(3, axis=2).copy()
    for _ in range(800):
        x, y = int(RNG.integers(0, SCENE_W)), int(RNG.integers(0, SCENE_H))
        color = tuple(int(c) for c in RNG.integers(0, 256, 3))
        kind = RNG.integers(0, 3)
        if kind == 0:
            cv2.circle(scene, (x, y), int(RNG.integers(4, 22)), color, -1)
        elif kind == 1:
            w, h = int(RNG.integers(8, 44)), int(RNG.integers(8, 44))
            cv2.rectangle(scene, (x, y), (x + w, y + h), color, -1)
        else:
            x2, y2 = x + int(RNG.integers(-40, 40)), y + int(RNG.integers(-40, 40))
            cv2.line(scene, (x, y), (x2, y2), color, int(RNG.integers(1, 4)))
    noise = RNG.integers(-8, 9, scene.shape)
    return np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def render_frames(scene: np.ndarray, out_dir: str) -> list[np.ndarray]:
    """Sweep a viewport left-to-right; save frame_*.jpg and return RGB frames."""
    xs = np.linspace(0, SCENE_W - VIEW_W, N_FRAMES)
    y_centre = (SCENE_H - VIEW_H) // 2
    frames = []
    for i, x in enumerate(xs):
        x = int(round(x))
        y = int(np.clip(y_centre + 25 * np.sin(2 * np.pi * i / N_FRAMES), 0, SCENE_H - VIEW_H))
        view = scene[y:y + VIEW_H, x:x + VIEW_W]
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
        frames.append(view.copy())
    return frames


def main() -> None:
    out = Path(__file__).parent / "videos"
    out.mkdir(exist_ok=True)
    scene = build_scene()

    with tempfile.TemporaryDirectory() as tmp:
        input_frames = render_frames(scene, tmp)
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
