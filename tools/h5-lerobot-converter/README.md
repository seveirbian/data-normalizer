# h5-lerobot-converter

将智元 A2D / 星海图 R1 机器人采集的具身操作数据（HDF5 格式）转换为 [LeRobot](https://github.com/huggingface/lerobot) 格式（parquet + mp4 + meta），支持 LeRobot 2.x 与 v3.0。

## 数据说明

| 字段 | 来源 | 维度 |
|------|------|------|
| `observation.state` | 双臂 + 夹爪 + 头部 + 腰部 + 底盘位置/朝向 | 27 |
| `action` | 双臂 + 夹爪 + 头部 + 腰部 + 底盘位置/速度指令 | 25 |
| `observation.images.hand_left` | 左手腕 RGB 相机 | 480×848 |
| `observation.images.hand_right` | 右手腕 RGB 相机 | 480×848 |
| `observation.images.head` | 头部 RGB 相机 | 720×1280 |
| `observation.images.head_depth` | 头部深度相机（PNG 16-bit，编码为 8-bit 灰度视频） | 720×1280 |

采集频率：30 fps

## 安装依赖

Python 依赖已统一在仓库根目录的 `pyproject.toml` 中管理，在仓库根目录执行：

```bash
uv sync
# 系统依赖
apt-get install -y ffmpeg
```

## 使用方法

在仓库根目录（或任意目录）通过 launcher 直接运行，无需安装、`cd` 或设置 `PYTHONPATH`。
第一个参数是转换模式：

| 模式 | 说明 | 对应模块 |
|------|------|----------|
| `a2d` | A2D HDF5 → LeRobot 2.x 格式 | `h5_lerobot_converter.convert` |
| `a2d-v3` | A2D HDF5 → LeRobot v3.0 格式 | `h5_lerobot_converter.convert_v3` |
| `r1-v3` | 星海图 R1 HDF5 → LeRobot v3.0 格式 | `h5_lerobot_converter.r1_convert_v3` |

模式之后的参数原样传给对应转换器。

### 单批次转换

```bash
python3 tools/h5-lerobot-converter/h5-lerobot-convert.py a2d \
  --input_dir /path/to/h5_files \
  --output_dir /path/to/lerobot_dataset \
  --task "pick up the cup and place it on the plate"
```

### 星海图 R1 → LeRobot v3.0 格式

```bash
python3 tools/h5-lerobot-converter/h5-lerobot-convert.py r1-v3 \
  --input_dir datasets/r1 \
  --output_dir /path/to/lerobot_dataset \
  --task "robot manipulation task" \
  --fps 30.0
```

参数与 `a2d-v3` 完全相同。R1 的 `observation.state` 和 `action` 均为 26 维（arm_position 12 + arm_velocity 12 + gripper 2），摄像头 5 路（双手 RGB+深度 + 头部 RGB）。

### 多批次追加（不同任务）

```bash
# 第一批：任务 A
python3 tools/h5-lerobot-converter/h5-lerobot-convert.py a2d \
  --input_dir /data/task_a \
  --output_dir /data/lerobot_dataset \
  --task "open the drawer"

# 第二批：任务 B，追加到同一数据集
python3 tools/h5-lerobot-converter/h5-lerobot-convert.py a2d \
  --input_dir /data/task_b \
  --output_dir /data/lerobot_dataset \
  --task "close the drawer"
```

### 等价调用方式

```bash
# 在工具目录内，作为模块运行：
cd tools/h5-lerobot-converter && python3 -m h5_lerobot_converter.convert --input_dir ... --output_dir ... --task ...

# 或把包目录加到 PYTHONPATH：
PYTHONPATH=tools/h5-lerobot-converter python3 -m h5_lerobot_converter.convert_v3 --input_dir ... --output_dir ... --task ...
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 包含 `.h5` 文件的目录 | 必填 |
| `--output_dir` | LeRobot 格式输出目录 | 必填 |
| `--task` | 任务描述（自然语言） | 必填 |
| `--fps` | 视频帧率 | 30.0 |
| `--chunk_size` | 每个 chunk 包含的 episode 数 | 1000 |

## 输出结构

```
output_dir/
  meta/
    info.json          # 数据集 schema、fps、episode 数等
    episodes.jsonl     # 每条轨迹的帧范围和任务描述
    tasks.jsonl        # 任务列表
    stats.json         # state/action 均值、方差（用于训练归一化）
  data/
    chunk-000/
      episode_000000.parquet   # 每条轨迹的表格数据（state/action/timestamp 等）
      episode_000001.parquet
  videos/
    chunk-000/
      observation.images.hand_left/episode_000000.mp4
      observation.images.hand_right/episode_000000.mp4
      observation.images.head/episode_000000.mp4
      observation.images.head_depth/episode_000000.mp4
```

## 运行测试

```bash
python3 -m pytest tests/ -v
```

## 视频编码说明

所有视频使用 `libx264 + yuv420p + crf 18 + -bf 0 + -g 1` 编码。

LeRobot 训练时 DataLoader 需要对视频做**随机帧访问**（每个 batch 从任意位置取帧），编码参数直接影响训练速度：

| 参数 | 值 | 原因 |
|------|-----|------|
| `-vcodec libx264` | H.264 | 通用硬件解码支持 |
| `-pix_fmt yuv420p` | YUV 4:2:0 | 兼容性最广 |
| `-crf 18` | 高质量 | 视觉损失极小 |
| **`-bf 0`** | **无 B-frames** | B-frames 解码需前后参考帧，随机访问触发大量 I/O，导致训练卡死 |
| **`-g 1`** | **每帧都是关键帧** | 默认 GOP=250 时随机 seek 需从上一 I-frame 逐帧解码，~0.5s/帧；`-g 1` 后 seek 变为 O(1)，~0.01s/帧，batch 取数据从 16s 降至 0.3s |

## LeRobot 训练建议

### 正式训练

```bash
uv run lerobot-train \
  --dataset.root=/path/to/lerobot_dataset \
  --dataset.repo_id=<your_repo_id> \
  --dataset.revision=v3.0 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --num_workers=0 \
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=100 \
  --output_dir=outputs/train/
```

### 快速冒烟测试（验证数据流通）

```bash
uv run lerobot-train \
  --dataset.root=/path/to/lerobot_dataset \
  --dataset.repo_id=<your_repo_id> \
  --dataset.revision=v3.0 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --num_workers=0 \
  --steps=20 \
  --save_freq=20 \
  --log_freq=1 \
  --output_dir=outputs/train_test/
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--num_workers=0` | 避免 CUDA + fork 死锁（torchcodec 在子进程中初始化 GPU 会卡死）|
| `--policy.push_to_hub=false` | 离线环境必须加，否则 checkpoint 保存时尝试推送到 HuggingFace 会卡住 |
| `--log_freq=1` | 测试时设为 1，否则默认 200 步才打一次日志，20 步训练看不到任何输出 |
| `--dataset.video_backend=pyav` | 可选，torchcodec 有问题时改用此 CPU 解码后端 |

## 注意事项

- **深度图**：原始格式保留，不做任何精度压缩。
- **多批次追加**：多次运行到同一 `output_dir` 会自动续接 episode 编号，stats 会重新计算。
- **任务描述**：同一批次内所有 `.h5` 文件共享一个任务描述。
- **重新转换**：修改视频编码参数后需删除旧数据集目录重新转换，同时清除 HF datasets 缓存：`rm -rf ~/.cache/huggingface/datasets/`。
