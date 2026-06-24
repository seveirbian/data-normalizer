# extrinsic-checker — 设计文档

**日期:** 2026-06-24
**状态:** 已批准(设计阶段)

## 目的

一个验证机器人数据集中**相机外参是否正确**的工具:用相机内参/外参 + URDF 正向运动学(FK),把每路相机看到的内容重建到机器人 base 坐标系,再检查重建结果在几何上是否合理。位于 `tools/extrinsic-checker/`。

包代码**与平台无关**;所有机器人/数据集相关的信息都放在通过 `--config` 传入的**外部自包含 JSON 配置**里。随工具提供一份 a2d 配置(`configs/a2d.json`),其内容来自前期已验证的关节映射与符号约定。

## 背景(a2d 配置所编码的已验证事实)

- 相机内参和外参来自 **h5 文件**(`parameters/camera/<cam>.json` 的 `intrinsic`、`extrinsic`)。配置文件**不**携带这些,只标明每路相机的挂载连杆和模态。
- 外参用法(已在 a2d head 相机上验证):JSON 里的 `E = [[R|t],[0,1]]` 把**相机光学系**的点变换到**挂载连杆系**,因此 `T_base_cam = T_base_link @ E`。相机光学系约定为 +x 右 / +y 下 / +z 前(RealSense / open3d 约定)。
- a2d 的 FK 注意点:h5 的 `joint_body_pitch` 需**取负**才匹配 URDF 的 `idx02_body_joint2`;base_link 前向轴是 `-x`。head 相机俯视桌面。
- a2d 相机:`head` 有 color+depth;`hand_left`/`hand_right` 只有 color。

## 两种验证方法(按相机模态自动选择)

**深度相机 → 点云反投法。**
用内参把 depth 反投成相机系点云,经 `T_base_cam` 变换到 base 系,RANSAC 拟合主平面,判定:
1. 主平面接近水平:`|n_z| >= plane_vertical_min`;
2. 平面高度落在 `table_height_range` 内;
3. 点云质心位于机器人前向一侧(由 `base_forward_axis` 决定)。
三条全满足则 PASS。产物:base 系彩色 PLY + 俯视正交投影 PNG。

**纯彩色相机 → 已知点投影法。**
对每个配置的目标连杆(如同臂夹爪),由 FK 取其在 base 系的原点,变换到相机系(`p_cam = E^-1 @ T_base_link^-1 @ p_base`),再用内参投影(`u = fx*X/Z + cx`,`v = fy*Y/Z + cy`)。判定:
1. 点在相机前方:`Z > 0`;
2. 像素落在图像内:`0 <= u < W 且 0 <= v < H`。
所有目标连杆都满足则 PASS。产物:在该相机 RGB 上叠加投影标记的 overlay PNG。

## 配置 schema(自包含 JSON)

```jsonc
{
  "urdf": "example-dataset/guodi/a2d/g1/g1_flat.urdf",  // 路径(先相对 config 文件目录解析,再相对 cwd)
  "base_link": "base_link",
  "base_forward_axis": "-x",                            // 取值 +x,-x,+y,-y 之一
  "joint_mapping": {
    "<组名>": {
      "h5_path": "joints/state/waist/position",
      "entries": [
        {"h5_index": 0, "urdf_joint": "idx02_body_joint2", "sign": -1},
        {"h5_index": 1, "urdf_joint": "idx01_body_joint1", "sign": 1}
      ]
    }
    // ... head、arm 等。(state/robot 浮动基不使用——所有检查都在 base_link 系内)
  },
  "cameras": {
    "head":       {"mount_link": "head_link2",     "modality": "depth"},
    "hand_left":  {"mount_link": "arm_l_end_link", "modality": "color",
                   "projection_targets": ["gripper_left_center_link"]},
    "hand_right": {"mount_link": "arm_r_end_link", "modality": "color",
                   "projection_targets": ["gripper_right_center_link"]}
  },
  "thresholds": {"plane_vertical_min": 0.85, "table_height_range": [0.3, 1.2]}
}
```

`sign` 缺省为 1。深度图的 `depth_scale` 固定为 1000.0(mm → m),这是这些 RealSense 流的标准。

## 架构

```
tools/extrinsic-checker/
  extrinsic_checker/
    loader.py            # 加载并校验 config;解析 urdf 路径
    kinematics.py        # yourdfpy 封装:读 h5 某帧关节,组装 cfg(映射+符号),取 T_base_link(link)
    depth_check.py       # 反投 + 变换 + RANSAC 平面判定;输出 PLY + 俯视 PNG
    projection_check.py  # 把目标连杆点投影到彩色图;输出 overlay PNG
    report.py            # Verdict 数据类 + 汇总 + 打印
    cli.py               # argparse: --config --input --camera --frame --out-dir;编排;退出码
    __main__.py
  extrinsic_check.py      # 独立启动器(任意 cwd 可跑,同 h5vid-export.py 风格)
  configs/
    a2d.json
  tests/
  README.md
  pyproject.toml          # 依赖: open3d, yourdfpy, h5py, opencv-python-headless, numpy
```

用 uv 环境(Python 3.12)运行,这些依赖已装好。

### 各组件职责

- **loader.py** — 解析 JSON 配置,校验必填键和枚举值(`base_forward_axis`、`modality`),解析 `urdf` 路径(先相对 config 文件目录,再相对 cwd),字段缺失/非法时报清晰错误。
- **kinematics.py** — 用 yourdfpy 加载一次 URDF。`build_cfg(h5, frame, joint_mapping)` 读取每组在 `frame` 处的数组、应用 `sign`,返回 `{urdf_joint: value}` 字典(未映射的关节默认 0)。`link_transform(cfg, link)` 返回 `T_base_link`(`get_transform(link, base_link)`)。
- **depth_check.py** — 从 h5 读该相机的内参 + depth 帧,用 open3d 反投,经 `T_base_cam = T_base_link @ E` 变换,RANSAC `segment_plane`,计算三条判据,返回 Verdict;写 base 系彩色 PLY 和俯视正交 PNG。
- **projection_check.py** — 读内参 + color 帧;对每个目标连杆由 FK 算 `p_base`,变换到相机系并投影;返回带逐目标 (u,v,Z,in_image) 明细的 Verdict;写 RGB overlay PNG。
- **report.py** — `Verdict` 数据类(camera、method、passed、metrics 字典、artifacts 列表);汇总并美化打印;给出总体通过/失败。
- **cli.py** — 解析参数;加载 config;打开 h5;对每个请求的相机按模态选方法运行;打印报告;任一相机 FAIL 或报错则退出码非 0。

## CLI

```
python3 tools/extrinsic-checker/extrinsic_check.py \
    --config configs/a2d.json \
    --input  FILE.h5 \
    [--camera head hand_left hand_right]   # 默认:config 里的全部相机
    [--frame 0]                            # 默认 0
    [--out-dir DIR]                        # 默认: ./extrinsic_check_out
```

逐相机打印:方法、指标、PASS/FAIL。仅当所有被检相机都通过时退出码为 0。

## 错误处理

- 配置字段缺失/非法,或 `base_forward_axis`/`modality` 取值未知 → 报错并指出字段。
- 请求的相机不在 config 中,或其模态数据在 h5 中不存在 → 报错并列出可用相机 / 现有模态。
- URDF 文件或被引用的 link/joint 找不到 → 报错并指出名称。
- 某帧的关节数组宽度小于映射的 `h5_index` → 报错。

## 测试(TDD)

纯函数单测(合成输入,不依赖大文件):
- **loader**:合法配置能解析;缺键 / 枚举非法时报错。
- **kinematics.build_cfg**:数组 + 映射 + 符号 → 期望的 cfg 字典(含符号取负、未映射关节默认 0)。
- **depth_check 判定**(`plane_verdict(points, thresholds, forward_axis)`):合成水平面 PASS;竖直面 FAIL;高度错误的平面 FAIL;质心在错误一侧 FAIL。
- **projection_check**(`project_point(p_base, T_base_link, E, fx,fy,cx,cy,W,H)`):单位变换下已知点投影到期望像素;相机后方的点(Z<0)FAIL;图像外的点 FAIL。

集成 / 冒烟(用真实文件,经 uv 运行):
- **FK**:加载 `g1_flat.urdf`,在 frame 0 套用 a2d head 配置,断言 `head_link2` 原点 ≈ `[0.495, 0, 1.385]`(容差 1 cm)。
- **a2d head 冒烟**:对真实 a2d h5 的 head 相机跑深度检查,断言 PASS 且 PLY + 俯视 PNG 已写出。
