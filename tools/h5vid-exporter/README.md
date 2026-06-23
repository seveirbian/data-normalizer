# h5vid-exporter

Render one or more camera image-stream topics from a guodi-style `.h5` file into
a single side-by-side (1×N) mp4 video.

## Topics

A *topic* is the full group path of an image stream, e.g.:

- `cameras/head/color`
- `cameras/head/depth`
- `cameras/hand_left/color`

Color streams are rendered as-is; depth streams are rendered with a JET colormap
(2nd–98th percentile normalization). Color vs depth is detected from the decoded
frame shape, not the topic name.

## Usage

From anywhere (e.g. the repo root) via the launcher — no install or `cd` needed:

```bash
python3 tools/h5vid-exporter/h5vid-export.py \
  --input /path/to/file.h5 \
  --topics cameras/head/color cameras/head/depth \
  --output out.mp4 \
  --fps 30 \
  --height 480
```

Equivalent alternatives:

```bash
# As a module, with the package dir on PYTHONPATH:
PYTHONPATH=tools/h5vid-exporter python3 -m h5vid_exporter --input ... --topics ... --output ...

# Or from inside the tool directory:
cd tools/h5vid-exporter && python3 -m h5vid_exporter --input ... --topics ... --output ...
```

Options:

- `--input` (required): input `.h5` file.
- `--topics` (required, 1+): full topic paths.
- `--output` (required): output `.mp4` path.
- `--fps`: frames per second, default `30`.
- `--height`: uniform per-tile height in pixels, default `480`.

If a topic is invalid, the tool prints the list of available topics in that file.

## Tests

```bash
python3 -m pytest -v
```
