#!/usr/bin/env python3
"""
center_ball.py

Take a folder of still images that each contain a ball (any type/color),
detect the ball in every image, then shift + uniformly scale each image so the
ball sits dead center at a consistent on-screen size. Output frames are a
portrait 9:16 canvas so they drop straight into a short-form vertical video,
regardless of whether the source images were portrait or landscape. Finally,
assemble the frames into a clip.

Detection: real footage has motion blur, seams/panels, varied ball colors, and
other balls in the background, all of which wreck a naive circle detector. So
we use a pretrained YOLO object detector and keep only the "sports ball" class
(COCO class 32), which matches balls by appearance regardless of color. When
several balls are visible we pick the one whose box center is closest to the
image center -- the subject ball is the framed one.

USAGE
-----
    python center_ball.py --input ./images --output ./out --debug

Common options:
    --width 1080 --height 1920   output canvas (default 1080x1920, i.e. 9:16)
    --ball-frac 0.20             target ball RADIUS as a fraction of canvas WIDTH
    --fps 12                     frame rate of the final clip
    --fill black|edge|blur       empty-border fill: black bars, mirrored edge,
                                 or a blurred full-bleed background (reels look,
                                 recommended). --edge-fill is a legacy alias.
    --shuffle [--seed N]         randomize frame order in the clip (seed to
                                 reproduce a given order)
    --debug                      write out/debug/ overlays showing detection
    --no-video                   just write centered frames, skip ffmpeg

RECOMMENDED FIRST RUN
---------------------
Run with --debug, then open out/debug/ and confirm the green circle lands on
the ball in every frame. Images where detection failed are listed at the end
and copied, un-centered, into out/failed/ so nothing disappears silently. If
some are missed, lower --conf or use a bigger --weights (see detect_ball()).

REQUIREMENTS
------------
    pip install opencv-python-headless numpy ultralytics  (see requirements.txt)
    ffmpeg on PATH (brew install ffmpeg)        -- only for the video step

The first run downloads the YOLO weights (~6 MB for yolov8n.pt) automatically.
"""

import argparse
import glob
import os
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np


BALL_CLASS = 32  # COCO class id for "sports ball"
_model = None


def load_model(weights="yolov8x.pt"):
    """Load (and cache) the YOLO detector. Import is lazy so the rest of the
    script still runs if ultralytics isn't installed until you need detection.
    The weights file auto-downloads on first use."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(weights)
    return _model


def detect_ball(img, model, conf=0.15, imgsz=1280, augment=False, device=None,
                min_area_frac=0.3):
    """Return (cx, cy, radius) of the subject sports ball, or None.

    Uses a pretrained YOLO detector and keeps only COCO class 32 ("sports
    ball"). This is robust to motion blur, ball color/panels, and background
    clutter that wreck a circle detector.

    Picking the right ball when several are visible: the subject ball is the
    one being played with, so it is BIG and near the camera, while other balls
    (players in the background) are small and distant. We therefore first drop
    every detection whose box is smaller than `min_area_frac` of the largest
    detection -- this removes the tiny, blurred background balls -- and only
    then choose the one closest to the image center among what survives. Using
    centrality alone was fragile: a small background ball that happened to sit
    near the frame center would win over the large subject ball in hand.
    Radius comes from the box size (mean half-side).

    Tuning if detection misbehaves (in order of impact on big photos):
      * imgsz: resolution detection runs at. These source photos are ~7000px;
        the default 640 shrinks them ~11x and the ball's detail is lost, so we
        raise it. 1280-1536 catches balls 640 misses, at the cost of speed.
      * weights (--weights / load_model): yolov8n is fastest but weakest;
        yolov8l/x recall far more, especially small or blurred balls.
      * conf: detection confidence floor. An unusually colored ball (e.g. a
        black/white one) is atypical for COCO and scores low, so 0.15 is a safe
        default; raise toward 0.3 for standard-colored balls to cut false hits.
      * min_area_frac: how much smaller than the biggest ball a detection may
        be before it's treated as background and ignored. LOWER -> keep smaller
        balls (risk grabbing a distant one), HIGHER -> only the dominant ball.
      * augment: test-time augmentation. Slower, squeezes out a few more.
    """
    h, w = img.shape[:2]
    res = model.predict(img, conf=conf, imgsz=imgsz, augment=augment,
                        device=device, classes=[BALL_CLASS], verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    area_gate = min_area_frac * float(areas.max())

    icx, icy = w / 2.0, h / 2.0
    best, best_d = None, float("inf")
    for (x1, y1, x2, y2), area in zip(xyxy, areas):
        if area < area_gate:
            continue  # small/distant ball -> background, skip
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r = ((x2 - x1) + (y2 - y1)) / 4.0
        d = (cx - icx) ** 2 + (cy - icy) ** 2  # squared dist to image center
        if d < best_d:
            best_d, best = d, (float(cx), float(cy), float(r))
    return best


def _cover_fill(img, out_w, out_h):
    """Scale img to COVER an out_w x out_h box (no letterbox) and center-crop
    it to exactly that size. Used to build a full-bleed background plate."""
    h, w = img.shape[:2]
    scale = max(out_w / w, out_h / h)
    rw, rh = int(np.ceil(w * scale)), int(np.ceil(h * scale))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    x0 = (rw - out_w) // 2
    y0 = (rh - out_h) // 2
    return resized[y0:y0 + out_h, x0:x0 + out_w]


def recenter(img, ball, out_w, out_h, target_radius, fill="black"):
    """Warp img into an out_w x out_h canvas with the ball centered & scaled.

    Uniform scale (same x and y) so the ball stays round. One warpAffine pass
    keeps interpolation clean. `fill` controls what goes in the empty area the
    warp leaves around the frame:
      * "black" -- solid black bars.
      * "edge"  -- mirror-pad the source across the border.
      * "blur"  -- a blurred, full-bleed copy of the source (the reels/Shorts
        look): the borders show plausible out-of-focus background instead of
        bars, and the sharp centered frame is composited on top.
    """
    cx, cy, radius = ball
    scale = target_radius / radius
    M = np.array([
        [scale, 0.0, out_w / 2.0 - scale * cx],
        [0.0, scale, out_h / 2.0 - scale * cy],
    ], dtype=np.float32)

    if fill != "blur":
        border = cv2.BORDER_REFLECT101 if fill == "edge" else cv2.BORDER_CONSTANT
        return cv2.warpAffine(
            img, M, (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=border,
            borderValue=(0, 0, 0),
        )

    # Blur fill: sharp centered frame over a blurred full-bleed background.
    fg = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ones = np.full(img.shape[:2], 255, np.uint8)
    mask = cv2.warpAffine(ones, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    bg = _cover_fill(img, out_w, out_h)
    sigma = out_w * 0.03  # blur strength scales with canvas size
    bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    bg = (bg * 0.85).astype(np.uint8)  # darken slightly so the subject pops

    # Feathered alpha composite for a seamless edge between sharp and blurred.
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)
    alpha = alpha[:, :, None]
    return (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Center a ball (any type/color) across images and build a vertical clip.")
    ap.add_argument("--input", required=True, help="folder of source images")
    ap.add_argument("--output", required=True, help="folder for results")
    ap.add_argument("--width", type=int, default=1080, help="output width (px)")
    ap.add_argument("--height", type=int, default=1920, help="output height (px)")
    ap.add_argument("--ball-frac", type=float, default=0.20,
                    help="target ball radius as a fraction of output WIDTH")
    ap.add_argument("--fps", type=float, default=12.0, help="clip frame rate")
    ap.add_argument("--fill", choices=["black", "edge", "blur"], default="black",
                    help="how to fill empty borders: black bars, mirror-padded "
                         "edge, or a blurred full-bleed background (reels look)")
    ap.add_argument("--edge-fill", action="store_true",
                    help="deprecated alias for --fill edge")
    ap.add_argument("--weights", default="yolov8x.pt",
                    help="YOLO weights (auto-downloads; yolov8n.pt is faster/weaker)")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="detection confidence floor (lower = more detections)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="detection resolution; raise for big photos (e.g. 1536)")
    ap.add_argument("--augment", action="store_true",
                    help="test-time augmentation: slower, catches a few more")
    ap.add_argument("--min-area-frac", type=float, default=0.3,
                    help="ignore balls smaller than this fraction of the "
                         "biggest detection (rejects distant background balls)")
    ap.add_argument("--device", default=None,
                    help="inference device: 0 (first GPU), cpu, etc. (default: auto)")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize frame order in the output clip")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for --shuffle (repeatable order)")
    ap.add_argument("--debug", action="store_true", help="write detection overlays")
    ap.add_argument("--no-video", action="store_true", help="skip ffmpeg assembly")
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp,tif,tiff,webp",
                    help="comma-separated image extensions to include")
    args = ap.parse_args()
    fill = "edge" if args.edge_fill else args.fill  # honor deprecated alias

    frames_dir = os.path.join(args.output, "frames")
    failed_dir = os.path.join(args.output, "failed")
    debug_dir = os.path.join(args.output, "debug")
    os.makedirs(frames_dir, exist_ok=True)

    exts = [e.strip().lower() for e in args.exts.split(",")]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.input, f"*.{e}"))
        files += glob.glob(os.path.join(args.input, f"*.{e.upper()}"))
    files = sorted(set(files))
    if not files:
        sys.exit(f"No images found in {args.input} (looked for: {', '.join(exts)})")

    if args.shuffle:
        random.Random(args.seed).shuffle(files)

    # Frames are numbered by success order below, so a shorter run must not
    # leave stale higher-numbered frames from a previous run in the sequence
    # (ffmpeg would fold them into the clip). Clear old frames first.
    for old in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(old)

    model = load_model(args.weights)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: using GPU {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA NOT available: running on CPU (install the CUDA build "
                  "of torch to use your GPU -- see requirements.txt).")
    except ImportError:
        pass
    target_radius = args.width * args.ball_frac
    failures, ok = [], 0

    print(f"Found {len(files)} images. Output {args.width}x{args.height} "
          f"(9:16 = {round(args.width/args.height, 4)}). "
          f"Target ball radius {target_radius:.0f}px.\n")

    for i, path in enumerate(files):
        name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] {name}: could not read")
            failures.append(path)
            continue

        ball = detect_ball(img, model, conf=args.conf, imgsz=args.imgsz,
                           augment=args.augment, device=args.device,
                           min_area_frac=args.min_area_frac)
        if ball is None:
            print(f"  [FAIL] {name}: no ball detected")
            failures.append(path)
            if args.debug:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{i:04d}_{name}"), img)
            continue

        cx, cy, r = ball
        out = recenter(img, ball, args.width, args.height,
                       target_radius, fill=fill)
        # Number frames by success order (contiguous 0000,0001,...) so gaps
        # from failed images never break ffmpeg's %04d sequence.
        cv2.imwrite(os.path.join(frames_dir, f"{ok:04d}.png"), out)
        ok += 1
        print(f"  [ ok ] {name}: {img.shape[1]}x{img.shape[0]} "
              f"ball@({cx:.0f},{cy:.0f}) r={r:.0f} -> centered")

        if args.debug:
            os.makedirs(debug_dir, exist_ok=True)
            dbg = img.copy()
            cv2.circle(dbg, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
            cv2.drawMarker(dbg, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 3)
            cv2.imwrite(os.path.join(debug_dir, f"{i:04d}_{name}"), dbg)

    if failures:
        os.makedirs(failed_dir, exist_ok=True)
        for p in failures:
            try:
                shutil.copy2(p, os.path.join(failed_dir, os.path.basename(p)))
            except OSError:
                pass

    print(f"\nDone: {ok} centered, {len(failures)} failed.")
    if failures:
        print(f"Failed images copied to: {failed_dir}")
        print("Tip: inspect out/debug/, then raise --imgsz (e.g. 1536), lower "
              "--conf (e.g. 0.1), or add --augment to catch the missed balls.")

    if args.no_video:
        return
    if ok == 0:
        print("No frames to assemble; skipping video.")
        return
    if shutil.which("ffmpeg") is None:
        print(f"\nffmpeg not on PATH -- frames are in {frames_dir}; "
              "install ffmpeg and assemble them, or use --no-video.")
        return

    out_video = os.path.join(args.output, "ball_centered.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_video,
    ]
    print(f"\nAssembling clip -> {out_video}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg failed:\n" + result.stderr[-1500:])
    else:
        print(f"Wrote {out_video}")


if __name__ == "__main__":
    main()