# bct-motion-track

Center a ball across a set of still photos and assemble them into a vertical (9:16) short-form clip.

Each source image contains a ball. The tool detects the ball, then shifts and uniformly scales every image so the ball sits dead center at a consistent size, and stitches the aligned frames into an MP4.

## How it works

- **Detection** — a pretrained [YOLOv8](https://github.com/ultralytics/ultralytics) object detector, keeping only the COCO `sports ball` class (id 32). Robust to motion blur, ball color/panels, and background clutter that break naive circle detectors.
- **Ball selection** — when several balls are visible, tiny/distant detections are dropped (`--min-area-frac`), then the one closest to the image center wins. The subject ball is big, near-camera, and framed.
- **Reframe** — one `warpAffine` pass centers and scales the ball onto a portrait canvas. Empty borders fill with black bars, mirrored edge, or a blurred full-bleed background (reels look).
- **Assembly** — `ffmpeg` builds the clip from the centered frames.

## Setup

Requires Python 3 and (for the video step) `ffmpeg` on PATH.

```bash
./setup.sh                          # creates .venv, installs requirements.txt
# or manually:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

YOLO weights auto-download on first run (`yolov8n.pt` ~6 MB, `yolov8x.pt` ~137 MB).

**GPU:** `ultralytics` pulls in `torch` (CPU-only by default). For an NVIDIA GPU, install the CUDA build first — see [requirements.txt](requirements.txt).

## Usage

```bash
python center_ball.py --input ./images --output ./out --debug
```

Recommended first run uses `--debug`: overlays land in `out/debug/` so you can confirm the green circle sits on the ball in every frame. Images where detection failed are copied un-centered into `out/failed/` — nothing disappears silently.

### Common options

| Option | Default | Meaning |
|--------|---------|---------|
| `--width` / `--height` | `1080` / `1920` | output canvas (9:16) |
| `--ball-frac` | `0.20` | target ball radius as fraction of canvas width |
| `--fps` | `12` | clip frame rate |
| `--fill black\|edge\|blur` | `black` | empty-border fill (`blur` = reels look) |
| `--weights` | `yolov8x.pt` | YOLO weights (`yolov8n.pt` faster/weaker) |
| `--conf` | `0.15` | detection confidence floor (lower = more hits) |
| `--imgsz` | `1280` | detection resolution; raise for big photos |
| `--min-area-frac` | `0.3` | ignore balls smaller than this fraction of the biggest |
| `--shuffle` `--seed N` | — | randomize frame order (seed to reproduce) |
| `--device` | auto | `0` (first GPU), `cpu`, etc. |
| `--no-video` | — | write frames only, skip ffmpeg |

Detection misses? Inspect `out/debug/`, then raise `--imgsz` (e.g. 1536), lower `--conf` (e.g. 0.1), or add `--augment`.

## Output

```
out/
  frames/            centered PNGs, numbered by success order
  failed/            images where no ball was detected (copied un-centered)
  debug/             detection overlays (with --debug)
  ball_centered.mp4  final clip
```

## Layout

- [center_ball.py](center_ball.py) — the whole tool (detect, reframe, assemble)
- [requirements.txt](requirements.txt) — Python deps + GPU notes
- [setup.sh](setup.sh) — one-time venv bootstrap
- `images/` — sample source photos
