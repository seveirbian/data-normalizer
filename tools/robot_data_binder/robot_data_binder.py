import json
import logging

import cv2
import numpy as np

from tools.robot_data_binder.data_reader.raw_data_load import HDF5Reader
from tools.robot_data_binder.robot_reader.robot_read import URDFReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)

# 每个关节可分别绑定的两路数据角色(采集时常同时有 state 与 action)
BINDING_ROLES = ("state", "action")


class RobotDataBinder:
    """把机器人描述文件(URDF)与原始数据(HDF5)绑定起来并回放校验。

    流程:
      load()    加载 URDF 和数据
      binding() 以 URDF 可驱动关节为 key,让使用者选择对应数据(路径+列号)为 value
      replay()  按映射逐帧驱动机器人并动画显示,供人工检查绑定是否正确
    """

    def __init__(self, urdf_path: str, data_path: str):
        self.urdf_path = urdf_path
        self.data_path = data_path
        self.robot_reader = URDFReader(urdf_path, load_meshes=True)
        self.data_reader = HDF5Reader(data_path)
        self.robot = None  # yourdfpy.URDF
        self.mapping = {}  # {joint_name: (h5_key, col)}
        self.cameras = {}  # 用于回放的相机映射 {自定义key: 数据路径(h5 key)}
        logger.info(
            "初始化 RobotDataBinder | urdf=%s | data=%s", urdf_path, data_path
        )

    def load(self):
        """加载机器人描述文件和数据。"""
        self.robot = self.robot_reader.load()
        self.data_reader.load()
        logger.info(
            "已加载 | 可驱动关节=%d | 数据集=%d",
            self.robot.num_actuated_joints,
            len(self.data_reader.index),
        )
        return self.robot, self.data_reader.index

    def binding(self, mapping=None):
        """建立 {可驱动关节: {"state": (数据路径,列号), "action": (数据路径,列号)}} 映射。

        每个关节可分别绑定 state 和 action 两路数据(采集时常同时有二者)。
        传入 mapping 则直接采用(便于脚本化);否则逐关节、逐角色交互式选择,
        回车可跳过某一路或整个关节(缺失的路在回放时按需选取)。
        """
        self._require_loaded()
        if mapping is not None:
            self.mapping = dict(mapping)
            logger.info("采用外部映射 | 已绑定 %d 个关节", len(self.mapping))
            return self.mapping

        self.mapping = {}
        keys = list(self.data_reader.index)
        print("可选数据:")
        self.data_reader.describe()
        for joint in self.robot.actuated_joint_names:
            entry = {}
            for role in BINDING_ROLES:
                picked = self._pick_dataset_col(
                    keys, f"关节 {joint} 的 {role} <- 数据序号(回车跳过)> "
                )
                if picked is not None:
                    entry[role] = picked
                    logger.info("绑定 %s.%s <- %s[:, %d]", joint, role, picked[0], picked[1])
            if entry:
                self.mapping[joint] = entry
        logger.info("binding 完成 | 已绑定 %d/%d 个关节", len(self.mapping), self.robot.num_actuated_joints)
        return self.mapping

    def _pick_dataset_col(self, keys, prompt):
        """交互式选一个 (数据路径, 列号);回车或无效序号返回 None。"""
        sel = input(prompt).strip()
        if not sel:
            return None
        try:
            key = keys[int(sel)]
        except (ValueError, IndexError):
            print("  无效序号,跳过")
            return None
        shape = self.data_reader.index[key]["shape"]
        col = 0
        if len(shape) >= 2:
            c = input(f"  列号 0..{shape[1] - 1}> ").strip()
            col = int(c) if c else 0
        return (key, col)

    def bind_camera(self, cameras=None):
        """指定用于回放的相机,存为 {自定义key: 数据路径} 映射(可多个)。

        传入 cameras(dict)则直接采用;否则先列出数据候选,由使用者反复输入
        一个相机 key 名、再为它选择数据序号,key 名留空结束。返回 {key: 数据路径} 映射。
        """
        self._require_loaded()
        if cameras is not None:
            for name, dpath in cameras.items():
                if dpath not in self.data_reader.index:
                    raise KeyError(f"相机 {name!r} 的数据路径不存在: {dpath}")
            self.cameras = dict(cameras)
            logger.info("指定相机 <- %s", self.cameras)
            return self.cameras

        keys = list(self.data_reader.index)
        print("可选数据:")
        self.data_reader.describe()
        self.cameras = {}
        while True:
            name = input("相机 key 名(回车结束)> ").strip()
            if not name:
                break
            sel = input(f"  {name} <- 数据序号> ").strip()
            try:
                self.cameras[name] = keys[int(sel)]
            except (ValueError, IndexError):
                print(f"  无效序号 {sel!r},跳过")
                continue
            logger.info("绑定相机 %s <- %s", name, self.cameras[name])
        logger.info("bind_camera 完成 | 已绑定 %d 个相机", len(self.cameras))
        return self.cameras

    def save_binding(self, path):
        """把当前映射关系(关节绑定 + 相机)保存为 JSON,与 load_binding 对称。"""
        data = {
            "urdf": self.urdf_path,
            "data": self.data_path,
            "mapping": {
                j: {role: [key, col] for role, (key, col) in roles.items()}
                for j, roles in self.mapping.items()
            },
            "cameras": dict(self.cameras),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            "已保存映射 -> %s | 关节=%d | 相机=%d",
            path,
            len(self.mapping),
            len(self.cameras),
        )
        return path

    def load_binding(self, path):
        """从 save_binding 保存的 JSON 恢复映射关系(关节绑定 + 相机)。

        与实例的 urdf/data 路径不一致时直接报错(绑定关系强依赖具体的 URDF 和数据)。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for field, cur in (("urdf", self.urdf_path), ("data", self.data_path)):
            if data.get(field) not in (None, cur):
                raise ValueError(
                    f"{field} 路径与当前不一致: 文件={data[field]} 当前={cur}"
                )
        self.mapping = {
            j: {role: (key, col) for role, (key, col) in roles.items()}
            for j, roles in data["mapping"].items()
        }
        self.cameras = dict(data.get("cameras", {}))
        logger.info(
            "已载入映射 <- %s | 关节=%d | 相机=%d",
            path,
            len(self.mapping),
            len(self.cameras),
        )
        return self.mapping

    def replay(self, role="state"):
        """按 binding 映射的 role(state 或 action)那一路逐帧驱动机器人并动画显示。需要图形界面。"""
        self._require_loaded()
        cols = self._role_cols(role)
        num_frames = min(len(c) for c in cols.values())
        logger.info("replay | role=%s | 帧数=%d | 绑定关节=%d", role, num_frames, len(cols))

        state = {"i": 0}

        def callback(scene):
            i = state["i"] % num_frames
            cfg = {j: 0.0 for j in self.robot.actuated_joint_names}
            for joint, col in cols.items():
                cfg[joint] = float(col[i])
            self.robot.update_cfg(cfg)
            state["i"] += 1

        self.robot.show(callback=callback)

    def replay_with_camera(self, role="state"):
        """在 replay(role: state 或 action)驱动机器人的同时,同步播放各相机视频。

        需要图形界面。相机视频用一个 matplotlib 多子图窗口,在机器人回放的逐帧
        回调里刷新;与关节按帧号同步,帧数取所有序列的较小值。
        """
        self._require_loaded()
        if not self.cameras:
            raise RuntimeError("未指定相机,请先调用 bind_camera()")

        # 预取绑定关节列 + 相机数据集
        cols = self._role_cols(role)
        cam_items = list(self.cameras.items())  # [(相机名, 数据路径), ...]
        cam_dsets = [self.data_reader.file[dpath] for _, dpath in cam_items]
        num_frames = min([len(c) for c in cols.values()] + [len(d) for d in cam_dsets])
        logger.info(
            "replay_with_camera | role=%s | 帧数=%d | 绑定关节=%d | 相机=%s",
            role,
            num_frames,
            len(cols),
            self.cameras,
        )

        import matplotlib.pyplot as plt

        plt.ion()
        fig, axes = plt.subplots(1, len(cam_dsets), figsize=(4 * len(cam_dsets), 4))
        axes = np.atleast_1d(axes)
        artists = []
        for ax, (name, _), dset in zip(axes, cam_items, cam_dsets):
            first = self._decode_frame(dset, 0)
            artists.append(ax.imshow(first, cmap=None if first.ndim == 3 else "viridis"))
            ax.set_title(name)
            ax.axis("off")
        fig.tight_layout()
        fig.show()

        state = {"i": 0}

        def callback(scene):
            i = state["i"] % num_frames
            cfg = {j: 0.0 for j in self.robot.actuated_joint_names}
            for joint, col in cols.items():
                cfg[joint] = float(col[i])
            self.robot.update_cfg(cfg)
            # 同步刷新各相机画面(手动泵 matplotlib 事件,与 pyglet 窗口并存)
            for art, dset in zip(artists, cam_dsets):
                art.set_data(self._decode_frame(dset, i))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            state["i"] += 1

        self.robot.show(callback=callback)
        plt.close(fig)

    def _role_cols(self, role):
        """预取每个关节在指定 role(state/action)下绑定的整列数据到内存,避免逐帧读 h5。"""
        if role not in BINDING_ROLES:
            raise ValueError(f"role 必须是 {BINDING_ROLES} 之一,收到 {role!r}")
        if not self.mapping:
            raise RuntimeError("映射为空,请先调用 binding()")
        cols = {}
        for joint, roles in self.mapping.items():
            if role not in roles:
                continue
            key, ci = roles[role]
            arr = self.data_reader.file[key]
            cols[joint] = arr[:, ci] if arr.ndim >= 2 else arr[:]
        if not cols:
            raise RuntimeError(f"没有任何关节绑定了 {role} 数据")
        return cols

    @staticmethod
    def _decode_frame(dset, i):
        """解码相机第 i 帧;彩色转 RGB 供 matplotlib 显示,深度等单通道原样返回。"""
        img = cv2.imdecode(np.asarray(dset[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _require_loaded(self):
        if self.robot is None:
            self.load()
        return self.robot
