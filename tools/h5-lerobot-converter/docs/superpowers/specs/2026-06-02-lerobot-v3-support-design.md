# LeRobot v3 Format Support Design

**Date:** 2026-06-02  
**Status:** Approved

## Goal

Add a parallel `convert_v3.py` entry point that converts A2D HDF5 robot data directly to LeRobot v3.0 format, alongside the existing v2 converter, sharing reader and video writer code.

## Approach

Direct HDF5 → v3 conversion (no intermediate v2 step, no dependency on the official migration tool). Two separate CLI entry points:

- `python3 -m h5_lerobot_converter.convert` — existing v2, untouched
- `python3 -m h5_lerobot_converter.convert_v3` — new v3

## Architecture

### Shared (unchanged)
- `h5_lerobot_converter/reader.py` — HDF5Reader → EpisodeData (state 27-dim, action 25-dim, 4-camera JPEG/PNG bytes)
- `h5_lerobot_converter/video_writer.py` — ffmpeg pipe writer, detects JPEG vs PNG, depth 16-bit→8-bit

### New files
- `h5_lerobot_converter/parquet_writer_v3.py` — accumulates rows across episodes, flushes when file size exceeds 100 MB
- `h5_lerobot_converter/meta_builder_v3.py` — writes tasks.parquet, episodes parquet, v3 info.json, stats.json
- `h5_lerobot_converter/convert_v3.py` — DatasetBuilderV3, CLI entry point
- `tests/test_convert_v3.py` — mirrors test_convert.py

## File Layout

```
output_dir/
  meta/
    info.json                                     # codebase_version: "v3.0"
    tasks.parquet                                 # task_index, task columns
    stats.json                                    # state/action mean/std/min/max
    episodes/
      chunk-000/
        file-000.parquet                          # one row per episode
  data/
    chunk-000/
      file-000.parquet                            # multi-episode, ≤100 MB
      file-001.parquet                            # overflow to next file
  videos/
    observation.images.hand_left/
      chunk-000/
        file-000.mp4                              # multi-episode video, ≤200 MB
    observation.images.hand_right/chunk-000/file-000.mp4
    observation.images.head/chunk-000/file-000.mp4
    observation.images.head_depth/chunk-000/file-000.mp4
```

## Data File Schema (per row)

Same as v2:
- `observation.state` — list[float32], length 27
- `action` — list[float32], length 25
- `timestamp` — float32, seconds within episode
- `frame_index` — int32, frame index within episode
- `episode_index` — int32, global episode index
- `index` — int32, global frame index
- `task_index` — int32
- `next.done` — bool, True on last frame of each episode

## Episodes Parquet Schema (per row = per episode)

- `episode_index` — int32
- `tasks` — list[str], e.g. `["pick up the cup"]`
- `length` — int32
- `dataset_from_index` — int32, first global frame index
- `dataset_to_index` — int32, last global frame index (exclusive)
- `data/chunk_index` — int32
- `data/file_index` — int32
- `videos/observation.images.hand_left/chunk_index` — int32
- `videos/observation.images.hand_left/file_index` — int32
- `videos/observation.images.hand_right/chunk_index` — int32
- `videos/observation.images.hand_right/file_index` — int32
- `videos/observation.images.head/chunk_index` — int32
- `videos/observation.images.head/file_index` — int32
- `videos/observation.images.head_depth/chunk_index` — int32
- `videos/observation.images.head_depth/file_index` — int32

## Tasks Parquet Schema

- `task_index` — int32
- `task` — str

## info.json (v3 additions)

```json
{
  "codebase_version": "v3.0",
  "robot_type": "agilex_a2d",
  "total_episodes": N,
  "total_frames": N,
  "fps": 30,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "splits": {"train": "0:N"},
  "features": { ... }
}
```

## Size-Based File Splitting

`ParquetWriterV3` maintains a current open buffer (list of row dicts). Before appending an episode's rows, it estimates the new file size (rough: bytes of raw float arrays + overhead). If estimated size would exceed `data_files_size_in_mb`, it flushes the current buffer to disk and opens a new file (`file_index += 1`). Video files use the same file index as the data file active when that episode was started.

## Multi-Batch Append

Running `convert_v3.py` twice on the same `output_dir` appends episodes. On startup, `DatasetBuilderV3._load_existing_state()` reads existing `meta/tasks.parquet` and `meta/episodes/.../file-*.parquet` to recover `tasks`, `episodes`, and `global_frame_index`. It then continues numbering episodes and files from where the previous run left off.

## CLI

```bash
python3 -m h5_lerobot_converter.convert_v3 \
  --input_dir /path/to/h5_files \
  --output_dir /path/to/dataset \
  --task "pick up the cup and place it on the plate" \
  [--fps 30.0] \
  [--chunk_size 1000] \
  [--data_file_size_mb 100] \
  [--video_file_size_mb 200]
```

## Error Handling

- Missing HDF5 keys: raised by `reader.py` (unchanged)
- NaN/Inf in state or action: stats.json validation catches at end
- File size estimation: conservative (over-estimate to avoid exceeding limits)

## Testing

`tests/test_convert_v3.py` covers:
1. Single episode produces correct file layout
2. `tasks.parquet` schema and content
3. `episodes/chunk-000/file-000.parquet` schema and content
4. `data/chunk-000/file-000.parquet` schema, row count, global indices
5. Video files exist and are non-empty
6. `info.json` has `codebase_version: "v3.0"`
7. `stats.json` has no NaN/Inf
8. Multi-batch append: episode count, frame continuity, task list
9. Size-based file splitting: second file created when first exceeds limit
