import os
import glob
import numpy as np
import cv2
from PIL import Image


def crop_letterbox(frames, var_threshold=5.0, int_threshold=10.0):
    """
    Detects and crops black margins (letterboxing) from a sequence of frames.

    This is critical for accurate Optical Flow. If black borders are present,
    feature detectors might latch onto the high-contrast
    edge between the image and the black border. Since the border doesn't move,
    this artificially biases the motion estimation toward zero, ruining the panorama.
    This happened in the Kessaria video during early testings

    Args:
        frames (list[np.ndarray]): List of RGB image frames.
        var_threshold (float): Variance threshold to determine if a column/row is "active" (contains image data).
        int_threshold (float): Intensity threshold to detect non-black pixels.

    Returns:
        list[np.ndarray]: A list of cropped frames.
    """
    if not frames:
        return frames

    h, w = frames[0].shape[:2]

    # optimize by only sampling a subset of frames to detect the border
    sample_indices = np.linspace(0, len(frames) - 1, min(25, len(frames)), dtype=int)
    sample_stack = np.array([frames[i] for i in sample_indices])

    # Calculate statistics across the sampled frames
    # We check if columns have variance (meaning image content changes) or high intensity
    col_variance = np.mean(np.std(sample_stack, axis=0), axis=(0, 2))
    col_max_val = np.max(np.max(sample_stack, axis=0), axis=(0, 2))
    is_active_col = (col_variance > var_threshold) | (col_max_val > int_threshold)

    # Scan from the center to the left to find the left margin
    center = w // 2
    left_margin = 0
    for col in range(center, -1, -1):
        # Look for a block of inactivity
        if not is_active_col[col] and not np.any(is_active_col[max(0, col - 5):col]):
            left_margin = col
            break

    # Scan from the center to the right to find the right margin
    right_margin = w
    for col in range(center, w):
        if not is_active_col[col] and not np.any(is_active_col[col:min(w, col + 5)]):
            right_margin = col
            break

    # Add a safety buffer to ensure no dark edge pixels remain
    buffer = 8
    left_margin = min(left_margin + buffer, center)
    right_margin = max(right_margin - buffer, center)

    # Crop all frames
    return [f[:, left_margin:right_margin, :] for f in frames]


def detect_orientation(frames):
    """
    Analyzes the dominant motion direction to distinguish between horizontal scanning
    and vertical scanning

    Handheld videos often start with a moment of stillness or jitter before the
    intended pan begins. Increasing the sample size ensures we capture the
    dominant panning motion rather than the initial handheld shake.
    """
    check_limit = min(30, len(frames))

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    total_dx, total_dy = 0.0, 0.0

    # Iterate through the checked frames
    for i in range(1, check_limit):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        # Detect features to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1500, qualityLevel=0.0001, minDistance=5)
        if p0 is None: continue

        # Calculate Optical Flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        # Accumulate absolute motion in X and Y directions
        if p1 is not None and len(p1[st == 1]) > 0:
            shifts = p1[st == 1] - p0[st == 1]
            total_dx += np.sum(np.abs(shifts[:, 0]))
            total_dy += np.sum(np.abs(shifts[:, 1]))
        prev_gray = curr_gray

    # Debug print to help you verify (optional, remove for final submission)
    print(f"Motion Analysis: dx={total_dx:.1f}, dy={total_dy:.1f}")

    # Heuristic: If vertical motion is 1.5x stronger than horizontal, assume vertical scan
    return total_dy > total_dx * 1.5

def generate_panorama_core(frames, n_out_frames):
    """
    Implements the core Manifold Mosaicing algorithm.

    It simulates a "Slit Camera" moving through space. By selecting different vertical
    slits from the input frames (Left side vs. Center vs. Right side), we can change
    the viewing angle of the resulting panorama, creating a stereo effect or
    a "wiggling" animation.

    Args:
        frames (list[np.ndarray]): The input video frames (RGB).
        n_out_frames (int): The number of frames in the output animation.

    Returns:
        list[PIL.Image]: The sequence of generated panoramic frames.
    """
    h, w = frames[0].shape[:2]
    num_frames = len(frames)

    # --- Step A: Compute Optical Flow ---
    raw_dx, raw_dy = [], []
    for i in range(1, num_frames):
        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        # Detect good features to track
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        dx, dy = 0.0, 0.0

        if p0 is not None:
            # Track features to next frame
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

            # Filter valid points and compute the median shift (robust to outliers)
            if p1 is not None and len(p1[st == 1]) > 0:
                shifts = p1[st == 1] - p0[st == 1]
                dx, dy = np.median(shifts[:, 0]), np.median(shifts[:, 1])

        raw_dx.append(dx)
        raw_dy.append(dy)

    # --- Step B: Motion Smoothing---
    # Handheld cameras often have jittery Y-axis motion.
    smooth_dx = np.convolve(raw_dx, np.ones(5) / 5, mode='same')
    smooth_dy = np.convolve(raw_dy, np.ones(5) / 5, mode='same')

    # --- Step C: Compute Global Alignment ---
    # We accumulate the relative shifts to get the global (absolute) position of each frame.
    full_acc_dx, full_acc_dy = np.zeros(num_frames), np.zeros(num_frames)
    for i in range(len(smooth_dx)):
        full_acc_dx[i + 1] = full_acc_dx[i] - smooth_dx[i]
        full_acc_dy[i + 1] = full_acc_dy[i] - smooth_dy[i]

    # Determine canvas size
    offset_x, offset_y = -np.min(full_acc_dx), -np.min(full_acc_dy)
    pano_w = int(np.ceil(np.max(full_acc_dx) - np.min(full_acc_dx) + w))
    pano_h = int(np.ceil(np.max(full_acc_dy) - np.min(full_acc_dy) + h))

    # --- Step D: Define Slit Trajectory ---
    # To create the "wiggle" stereo effect, we move the sampling slit back and forth
    # across the source image.
    # Left side of source frame -> View from the right (Stereo Right)
    # Right side of source frame -> View from the left (Stereo Left)
    slit_margin = w // 4
    half = n_out_frames // 2
    l_to_r = np.linspace(slit_margin, w - slit_margin, half)
    slit_positions = np.concatenate([l_to_r, np.flip(l_to_r)])

    generated_frames = []

    # --- Step E: Generate Each Frame of the Output Animation ---
    for slit_x in slit_positions[:n_out_frames]:
        # Use float32 for high-precision blending accumulation
        canvas = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((pano_h, pano_w, 1), dtype=np.float32)

        for i in range(num_frames):
            # Calculate strip width dynamically based on speed.
            # Faster camera motion = wider strips needed to avoid gaps.
            d = abs(smooth_dx[i - 1]) if i > 0 else 10
            hw = max(15, int(d) + 10)  # Half-width

            # Extract the strip from the current frame based on the slit position
            l_b, r_b = max(0, int(slit_x - hw)), min(w, int(slit_x + hw))
            strip = frames[i][:, l_b:r_b, :].astype(np.float32)

            # Create a linear Alpha Mask for "Feathering"
            # This blends the edges of the strip so we don't see hard cut lines in the mosaic.
            sw = strip.shape[1]
            mask = np.ones((h, sw, 1), dtype=np.float32)
            feather = min(5, sw // 8)
            mask[:, :feather] = np.linspace(0, 1, feather).reshape(1, -1, 1)
            mask[:, -feather:] = np.linspace(1, 0, feather).reshape(1, -1, 1)

            # Calculate where this strip belongs on the global canvas
            # We align the *center* of the strip (slit_x) to the global trajectory
            px = int(np.round(full_acc_dx[i] + offset_x + (w // 2) - (sw // 2)))
            py = int(np.round(full_acc_dy[i] + offset_y))

            # Boundary checks to ensure we stay inside the canvas
            y1, y2 = max(0, py), min(pano_h, py + h)
            x1, x2 = max(0, px), min(pano_w, px + sw)

            if y1 < y2 and x1 < x2:
                # Add the weighted strip to the canvas
                s_part = strip[y1 - py:y1 - py + (y2 - y1), x1 - px:x1 - px + (x2 - x1)]
                m_part = mask[y1 - py:y1 - py + (y2 - y1), x1 - px:x1 - px + (x2 - x1)]
                canvas[y1:y2, x1:x2] += s_part * m_part
                weight_sum[y1:y2, x1:x2] += m_part

        # Normalize the canvas by the weights to average overlapping pixels
        # (Avoids division by zero with a small epsilon)
        canvas /= (weight_sum + 1e-6)

        # Convert back to uint8 Image
        generated_frames.append(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)))

    return generated_frames


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for Exercise 4.

    Reads a sequence of frames from a directory, determines the correct orientation,
    and generates a stereo panoramic video (list of images).

    Args:
        input_frames_path (str): Path to a directory containing frames named 'frame_00000.jpg', etc.
        n_out_frames (int): Number of frames in the output stereo animation.

    Returns:
        list[PIL.Image]: A list of PIL images representing the generated video frames.
    """
    # Load frames
    search_path = os.path.join(input_frames_path, "frame_*.jpg")
    files = sorted(glob.glob(search_path))
    if not files:
        raise ValueError(f"No frames found in {input_frames_path}. Expected format: frame_*.jpg")

    print(f"Loading {len(files)} frames from {input_frames_path}...")
    frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]

    # Pre-process: Remove letterbox borders
    frames = crop_letterbox(frames)

    # Pre-process: Check for vertical orientation (e.g., Trees video)
    is_vertical = detect_orientation(frames)
    if is_vertical:
        print("Vertical motion detected. Rotating frames 90 degrees.")
        frames = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
    else:
        print("Horizontal motion detected.")

    # Core Algorithm
    result = generate_panorama_core(frames, n_out_frames)

    # Post-process: Rotate back if necessary
    if is_vertical:
        return [Image.fromarray(cv2.rotate(np.array(p), cv2.ROTATE_90_COUNTERCLOCKWISE)) for p in result]

    return result
