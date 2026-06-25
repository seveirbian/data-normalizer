# lerobot-cam-pose-adder — 设计文档

**日期:** 2026-06-25
**状态:** 已批准(设计阶段)

## 目的

给一个已从 h5 转换好的 **LeRobot v3** 数据集,逐帧计算每个相机在 **base_link** 系下的位姿,并作为一个新 feature `observation.camera_poses` 写入数据集副本。位姿 = 关节状态经 URDF 正向运动学 + 已验证的相机外参得到。

这次工作分两部分:
1. **重构**:把 extrinsic-checker 里被复用的核心(配置加载、FK、相机参数)抽到公共包 `tools/common/robotkit/`,extrinsic-checker 改为依赖它(行为不变,测试保持通过)。
2. **新工具** `tools/lerobot-cam-pose-adder/`,基于 `robotkit` 实现。

## 输入(全部已就位)

- LeRobot v3 数据集目录(`meta/` + `data/` + `videos/`)。
- `tools/configs/a2d.json`:URDF、base_link、`joint_mapping`(h5→urdf + 符号)、`cameras`(挂载连杆 + `extrinsic`)。
- `tools/configs/a2d_h5_lerobot_map.json`:逐维 h5↔lerobot 字段映射,用于把 `observation.state` 的列定位到 `(h5_path, h5_index)`。

**本工具不读原始 h5**:关节值从 LeRobot `observation.state` 取,外参从 config 取。

## 第一部分:公共包 `robotkit`

位置 `tools/common/robotkit/`,作为各工具共享的库。各工具的启动器与 `conftest.py` 把 `tools/common` 加入 `sys.path`,从而 `import robotkit`。

公共 API(从 extrinsic_checker 平移,不改语义):
- `robotkit.config.load_config(path)` — 加载并校验 config,解析 urdf 路径(原 `loader.py`)。
- `robotkit.kinematics.Kinematics(urdf, base_link).link_transform(cfg, link)`、`build_cfg(frame_values, joint_mapping, actuated_joints)`(原 `kinematics.py`)。
- `robotkit.camera.extrinsic_matrix(extr)`、`resolve_extrinsic(cam_cfg, h5, cam)`、`read_intrinsic`、`read_extrinsic`(原 `h5read.py` 的相机参数部分)。
- `robotkit.h5io.open_h5 / decode_image / has_modality / read_frame_values`(原 `h5read.py` 的 h5 I/O 部分)。

extrinsic-checker 重构后只保留自身逻辑(`depth_check`、`projection_check`、`report`、`cli`),从 `robotkit` import 上述能力。其现有 24 个测试保持通过(import 路径更新)。

## 第二部分:`lerobot-cam-pose-adder`

### 数据流(逐帧)

```
observation.state[t] (按 a2d.json joint_mapping × 映射文件 链接取关节值)
   └─ 对每个 joint_mapping entry: (h5_path, h5_index)
        ──[映射文件: observation.state 组件,按 (h5_path,h5_index) 命中]──▶ lerobot_index
        cfg[urdf_joint] = state[t][lerobot_index] × sign
   ──robotkit.Kinematics FK──▶ T_base_link(mount_link)        # 每相机
   T_base_cam = T_base_link @ E                                # E = extrinsic_matrix(config 的 extrinsic)
   pose = [x,y,z, qx,qy,qz,qw]                                 # 四元数 xyzw(scipy as_quat,与 base_ori 一致)
camera_poses[t] = head(7) ++ hand_left(7) ++ hand_right(7)     # 21 维
```

相机顺序固定:`head, hand_left, hand_right`。

### 写回(`--output` 新数据集副本)

1. 把整个数据集复制到 `--output`(`meta/` + `data/` + `videos/` 原样复制)。原数据集完全不动。
2. 在副本的 `data/**/*.parquet` 每个文件加一列 `observation.camera_poses`(每行 21 维 float32),按 `frame_index`/`index` 对齐。
3. 副本 `meta/info.json`:加 feature `observation.camera_poses`(`dtype=float32`、`shape=[21]`、`names`=每相机 `{cam}_x,{cam}_y,{cam}_z,{cam}_qx,{cam}_qy,{cam}_qz,{cam}_qw`,按相机顺序)。
4. 副本 `meta/stats.json`:为该 feature 写 `mean/std/min/max`(逐维 21)、`count`([N])、`q01/q10/q50/q90/q99`(逐维 21),跨所有帧统计,结构对齐现有项。

### 模块结构

```
tools/lerobot-cam-pose-adder/
  lerobot_cam_pose_adder/
    fieldmap.py     # 加载 h5↔lerobot 映射;build (h5_path,h5_index)->lerobot_index 查找表
    poses.py        # state_row+config 链接成 cfg;mat_to_pose(4x4->7);camera_poses_row(cfg,E,cameras)->21
    dataset.py      # 复制数据集;改写 parquet 加列;更新 info.json/stats.json
    cli.py          # argparse: --dataset --config --fieldmap --output [--overwrite]; 编排所有帧
    __main__.py
  lerobot_cam_pose_adder.py   # 启动器(任意 cwd;sys.path 加 tool 目录 + tools/common)
  tests/
  README.md
  pyproject.toml
```

依赖(uv 环境已具备):`robotkit` 所需的 yourdfpy/numpy + `pandas`/`pyarrow`(读写 parquet)+ `scipy`(四元数)。

### CLI

```
uv run python tools/lerobot-cam-pose-adder/lerobot_cam_pose_adder.py \
    --dataset  example-dataset/guodi/a2d/lerobotV3 \
    --config   tools/configs/a2d.json \
    --fieldmap tools/configs/a2d_h5_lerobot_map.json \
    --output   <新数据集目录>
```

若 `--output` 已存在或源已含 `observation.camera_poses` → 报错(除非 `--overwrite`)。

## 错误处理

- config/fieldmap 字段缺失或非法 → 指明字段报错。
- joint_mapping 的 `(h5_path, h5_index)` 在 fieldmap 的 `observation.state` 中找不到 → 报错列出该条。
- URDF / 挂载连杆 / 相机缺失 → robotkit 给出清晰错误。
- `--output` 已存在且无 `--overwrite` → 报错。

## 测试(TDD)

**robotkit**:沿用 extrinsic-checker 现有纯函数/集成测试,import 路径改为 `robotkit.*`,全部保持通过。

**lerobot-cam-pose-adder 纯函数**:
- `mat_to_pose`:已知 4×4(平移+绕轴旋转)→ 期望 `[x,y,z,qx,qy,qz,qw]`(四元数数值核对)。
- `fieldmap` 查找:`(h5_path,h5_index)` → 正确 `lerobot_index`;缺失则报错。
- `state_to_cfg`:给定 state 行 + a2d joint_mapping + fieldmap → cfg 字典,核对 `joint_body_pitch` 被取负、其它正确、未映射关节为 0。
- `camera_poses_row`:给定 cfg + 每相机 E → 21 维,长度/分段正确。

**集成**(真实 a2d lerobotV3,经 uv):
- 跑完生成 `--output`:存在 `observation.camera_poses`(shape 21),`info.json`/`stats.json` 已更新,原数据集字节不变。
- **交叉校验**:cam-pose-adder 在 frame 0 算出的 head 位姿,与 extrinsic-checker 用同一 frame 的 `T_base_cam`(FK∘E)在数值上一致(平移 + 旋转,容差 1e-4)。这把"从 lerobot state 重建"和"从 h5 重建"对齐,确认链接逻辑正确。
