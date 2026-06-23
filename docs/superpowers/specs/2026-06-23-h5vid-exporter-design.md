# h5vid-exporter — Design

**Date:** 2026-06-23
**Status:** Approved (design phase)

## Purpose

A formal tool module that, given one h5 file and one or more camera image-stream
topics, renders all topics side by side into a single video file. Lives at
`tools/h5vid-exporter/`.

This is the first real module in the (currently greenfield) `data-normalizer`
project, so it also bootstraps project dependencies.

## Scope

In scope:
- Camera **image streams only** (color and depth). A "topic" is a full h5 group
  path such as `cameras/head/color` or `cameras/head/depth`.
- Output: one mp4 video with all requested topics composited in a single
  horizontal row (1×N).

Out of scope (YAGNI):
- Numeric data (joints, timestamps) rendered as plots.
- Arbitrary non-image datasets.
- Topic shorthand resolution — only full paths are accepted.
- Reading fps or any timing from `metadata.json`.

## Data background

The guodi h5 files store camera frames as **encoded jpg/png bytes** inside
`object`-dtype datasets at `<topic>/data`, shape `(num_frames,)`. Color streams
decode to 3-channel images; depth streams decode to single-channel (uint16)
images. Generic h5 viewers cannot render these, which is why this tool exists.

## CLI

```
h5vid-export --input FILE.h5 \
             --topics cameras/head/color cameras/head/depth \
             --output out.mp4 \
             [--fps N]      # default 30, pure CLI control, never read from file
             [--height N]   # uniform per-tile height, default 480
```

- `--input` (required): path to the h5 file.
- `--topics` (required, 1+): full group paths to image streams.
- `--output` (required): output mp4 path.
- `--fps` (optional): default 30.
- `--height` (optional): default 480.

## Architecture

Directory layout under `tools/h5vid-exporter/`:

```
h5vid_exporter/
  reader.py     # open h5, validate topic paths, read per-frame bytes, frame counts
  render.py     # bytes -> RGB tile; color/depth decision by decoded shape; label; resize
  compose.py    # hstack same-frame tiles into one 1xN frame
  exporter.py   # orchestration: iterate frames, build rows, write mp4
  cli.py        # argparse entry point
tests/          # unit tests using a synthetic tiny h5 fixture
README.md
```

Dependencies added to the project: `h5py`, `numpy`, `opencv-python`,
`imageio[ffmpeg]`.

### Component responsibilities

**reader.py**
- Opens the h5 file.
- Validates each requested topic: the path must exist and resolve to an image
  stream (a `<topic>/data` dataset). On failure, raise an error that lists the
  available camera topics in that file (no silent failure).
- Provides access to encoded frame bytes for a given topic and frame index.
- Reports each topic's frame count.

**render.py**
- Decodes jpg/png bytes to an image array (opencv `imdecode`).
- Decides color vs depth **by the decoded array shape**, not by the path name:
  - 3-channel → color, convert BGR→RGB.
  - single-channel / uint16 → depth: normalize by the 2nd–98th percentile of
    valid (>0) values, then apply a JET colormap.
- Draws a label bar with the topic path at the top of the tile.
- Resizes the tile to the common target height (preserving aspect ratio).

**compose.py**
- Takes the list of rendered tiles for one frame index and horizontally stacks
  them (1×N). Tiles already share a height; widths may differ.

**exporter.py**
- Orchestrates: determines the frame range (see Frame alignment), and for each
  frame index reads + renders + composes all topics, then writes the frame via
  the encoder.
- Encoding uses `imageio` with the ffmpeg backend (libx264) for reliable mp4
  output independent of system codecs.

**cli.py**
- Parses arguments and invokes the exporter.

## Key behavior decisions

1. **color/depth detection by decoded shape** — robust against naming
   variations across platforms (`hand_left` vs `left_hand`, etc.).
2. **Frame alignment** — use the **minimum** frame count across all requested
   topics. If counts differ, emit a warning (the dataset is already aligned, so
   this is a safety net).
3. **Topic validation** — on a missing or non-image topic, raise an error that
   lists the file's available camera topics.
4. **Label bar** — each tile is annotated with its topic path.
5. **fps** — taken purely from `--fps` (default 30); the file is never read for
   timing.

## Error handling

- Missing/invalid topic → error listing available camera topics.
- Topic exists but bytes fail to decode → error identifying the topic and frame.
- Frame-count mismatch across topics → warning, proceed with the minimum count.
- Missing output directory → created before writing, or a clear error.

## Testing (TDD)

Build a synthetic tiny h5 fixture (2–3 frames; one color topic encoded as jpg
bytes, one depth topic encoded as single-channel png bytes) and cover:

- **reader**: topic path validation (valid path passes; invalid path raises and
  lists available topics); correct frame counts.
- **render**: color branch (BGR→RGB, 3-channel out) and depth branch
  (single-channel → 3-channel JET); label bar drawn; output height equals target.
- **compose**: hstacking N tiles yields the expected width/height.
- **exporter**: end-to-end produces a readable mp4 with the expected frame count
  and dimensions; frame-count mismatch triggers the minimum-count path.
