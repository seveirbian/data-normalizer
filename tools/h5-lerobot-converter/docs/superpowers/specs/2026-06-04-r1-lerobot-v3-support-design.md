# 星海图 R1 机器人数据转 LeRobot v3 格式 — 设计文档

**日期：** 2026-06-04  
**范围：** 新增 `r1_reader.py` + `r1_convert_v3.py`，支持将星海图 R1 HDF5 数据转换为 LeRobot v3.0 格式

---

## 背景

项目已有 A2D → LeRobot v3 的转换链路。R1 与 A2D 使用相同的 HDF5 容器格式（`metadata.json ver 2.1.0`），差异仅在关节结构和摄像头配置上。后端的 `ParquetWriterV3`、`MetaBuilderV3`、`VideoWriter` 均可直接复用，只需新增一个 R1 专用 reader 和一个薄封装的 CLI。

---

## R1 数据格式

### 关节

| 路径 | 维度 | 说明 |
|---|---|---|
| `joints/state/arm/position` | (N, 12) | 左右臂各 6 关节位置 |
| `joints/state/arm/velocity` | (N, 12) | 左右臂各 6 关节速度 |
| `joints/state/effector/position` | (N, 2) | 左右夹爪位置 |
| `joints/action/arm/position` | (N, 12) | 同上，动作指令 |
| `joints/action/arm/velocity` | (N, 12) | 同上，动作指令 |
| `joints/action/effector/position` | (N, 2) | 同上，动作指令 |

`observation.state` = `arm/position(12)` + `arm/velocity(12)` + `effector/position(2)` = **26 维**  
`action` = `arm/position(12)` + `arm/velocity(12)` + `effector/position(2)` = **26 维**

关节名称顺序：`left_arm_joint1~6, right_arm_joint1~6`（position/velocity 各 12），`left_gripper, right_gripper`（2）。

### 摄像头（5 路）

| LeRobot key | HDF5 路径 | 分辨率 |
|---|---|---|
| `observation.images.hand_left` | `cameras/hand_left/color/data` | 480×640 RGB |
| `observation.images.hand_left_depth` | `cameras/hand_left/depth/data` | 480×640 灰度 |
| `observation.images.hand_right` | `cameras/hand_right/color/data` | 480×640 RGB |
| `observation.images.hand_right_depth` | `cameras/hand_right/depth/data` | 480×640 灰度 |
| `observation.images.head` | `cameras/head/color/data` | 360×640 RGB |

### 时间戳

`timestamp` 字段为纳秒整数，除以 `1e9` 转为秒（与 A2D 相同）。

---

## 架构

```
r1_reader.py          新建   读取 R1 HDF5，返回 EpisodeData（26 维 state/action，5 路摄像头）
r1_convert_v3.py      新建   DatasetBuilderR1V3 + CLI，复用 ParquetWriterV3 / MetaBuilderV3
tests/test_r1_reader.py       新建   R1Reader 单元测试
tests/test_r1_convert_v3.py   新建   R1 → LeRobot v3 集成测试
```

**不修改的现有文件：** `reader.py`、`convert_v3.py`、`parquet_writer_v3.py`、`meta_builder_v3.py`、`video_writer.py`

---

## 组件设计

### `r1_reader.py`

```python
class R1Reader:
    def read(self, path: Path) -> EpisodeData:
        ...
```

- 拼接 `state`：`state/arm/position` + `state/arm/velocity` + `state/effector/position` → float32 (N, 26)
- 拼接 `action`：`action/arm/position` + `action/arm/velocity` + `action/effector/position` → float32 (N, 26)
- 时间戳：`timestamp / 1e9` → float32 (N,)
- 图像：读取 5 路摄像头的 JPEG 字节序列

复用现有 `EpisodeData` dataclass（字段含义通用，注释中的 shape 仅是 A2D 的例子，不影响 R1 使用）。

### `r1_convert_v3.py`

`DatasetBuilderR1V3` 与 `DatasetBuilderV3` 结构完全相同，差异：
- 用 `R1Reader` 替换 `HDF5Reader`
- `CAMERA_KEYS` 改为 R1 的 5 路摄像头键名

CLI 入口：`python3 -m h5_lerobot_converter.r1_convert_v3`，参数与 `convert_v3` 相同。

### 元数据

`meta/info.json` 中：
- `robot_type: "r1"`
- `features.observation.state.shape: [26]`
- `features.action.shape: [26]`
- 5 路摄像头的 video feature 条目

---

## 测试策略

- `test_r1_reader.py`：使用 `conftest.py` 的 fixture 模式，构造最小 R1 HDF5，验证 state/action shape、时间戳转换、5 路图像读取
- `test_r1_convert_v3.py`：端到端集成测试，验证输出文件结构、info.json 内容、parquet schema、stats 无 NaN

---

## CLI 用法

```bash
python3 -m h5_lerobot_converter.r1_convert_v3 \
  --input_dir datasets/r1 \
  --output_dir /path/to/output \
  --task "robot manipulation task" \
  --fps 30.0
```
