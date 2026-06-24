# A2D HDF5 → LeRobot 3.0 转换脚本设计

## 背景

智元 A2D 机器人采集的具身操作数据存储为 HDF5 格式（ver 2.1.0），需要转换为 LeRobot 3.0 格式以供训练使用。每个 `.h5` 文件对应一条 episode，30fps，约 2138 帧。

## 目标

- 独立 Python 脚本，不依赖 lerobot 库
- 批量转换目录下所有 `.h5` 文件
- 支持用户指定任务描述（per-batch）
- 支持多批次（多次运行，每次不同 task）
- 原始数据全量保留，不裁剪任何字段

## 字段映射

### observation.state（27 维，float32）

| 索引 | 来源路径 | 含义 |
|------|----------|------|
| 0–13 | joints/state/arm/position | 双臂关节位置 |
| 14–15 | joints/state/effector/position | 双手夹爪 |
| 16–17 | joints/state/head/position | 头部关节 |
| 18–19 | joints/state/waist/position | 腰部关节 |
| 20–22 | joints/state/robot/position | 底盘位置 xyz |
| 23–26 | joints/state/robot/orientation | 底盘朝向四元数 |

### action（25 维，float32）

| 索引 | 来源路径 | 含义 |
|------|----------|------|
| 0–13 | joints/action/arm/position | 双臂关节指令 |
| 14–15 | joints/action/effector/position | 双手夹爪指令 |
| 16–17 | joints/action/head/position | 头部关节指令 |
| 18–19 | joints/action/waist/position | 腰部关节指令 |
| 20–22 | joints/action/robot/position | 底盘位置指令 |
| 23–24 | joints/action/robot/velocity | 底盘速度指令 |

### 相机（4路视频）

| LeRobot 特征名 | HDF5 路径 | 分辨率 |
|----------------|-----------|--------|
| observation.images.hand_left | cameras/hand_left/color/data | 480×848 RGB |
| observation.images.hand_right | cameras/hand_right/color/data | 480×848 RGB |
| observation.images.head | cameras/head/color/data | 480×848 RGB |
| observation.images.head_depth | cameras/head/depth/data | 480×848 depth |

## 架构

```
convert_h5_lerobot_converter.py
  HDF5Reader     — 读取单个 .h5，返回帧数据
  VideoWriter    — ffmpeg pipe，将 jpg bytes 流编码为 mp4
  ParquetWriter  — 写单条 episode 的 parquet 文件
  DatasetBuilder — 协调转换流程，生成 meta 文件
  main()         — CLI 入口
```

## 输出格式（LeRobot 3.0）

```
output_dir/
  meta/
    info.json          # 特征 schema、fps、总帧数、总 episode 数
    episodes.jsonl     # 每条 episode 的帧范围和 task_index
    tasks.jsonl        # 任务描述列表
    stats.json         # 各特征均值/方差（用于训练归一化）
  data/
    chunk-000/
      episode_000000.parquet
      ...
  videos/
    chunk-000/
      observation.images.hand_left/episode_000000.mp4
      observation.images.hand_right/episode_000000.mp4
      observation.images.head/episode_000000.mp4
      observation.images.head_depth/episode_000000.mp4
```

## CLI

```bash
python convert_h5_lerobot_converter.py \
  --input_dir /root/datasets/guodi/a2d \
  --output_dir /root/datasets/lerobot/task1 \
  --task "pick up the apple" \
  [--chunk_size 1000]
```

## 处理流程

1. 扫描 `input_dir`，收集所有 `.h5` 文件并排序
2. 逐文件处理：
   a. 读取 HDF5，提取 state/action/timestamp/图像
   b. 拼接 state（27维）和 action（25维）为 float32
   c. 写 parquet（每行一帧，包含 frame_index、episode_index、index、task_index、timestamp、next.done）
   d. 4 路视频通过 ffmpeg stdin pipe 编码为 mp4
3. 所有文件处理完毕后：
   a. 写 tasks.jsonl / episodes.jsonl / info.json
   b. 遍历所有 parquet 计算 stats.json（均值、方差、min、max）

## 依赖

- h5py
- numpy
- pandas + pyarrow（parquet 读写）
- ffmpeg（命令行工具，视频编码）
- cv2（可选，调试用）
