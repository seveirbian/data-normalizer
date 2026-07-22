import json
import logging
from abc import ABC, abstractmethod

import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)

# 面向具身数据的可视化种类:未确认一律 raw 兜底,由使用者显式标注其余三类
KINDS = ("raw", "image", "series", "text")


class RawDataLoader(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load(self):
        """读取原始数据,解析成内部数据结构"""
        ...

    @abstractmethod
    def render(self, key):
        """按 key 已标注的种类渲染对应内容"""
        ...


class HDF5Reader(RawDataLoader):
    """通用 HDF5 原始数据读取/浏览器。

    不预设任何 key 规范,也不猜测语义:load() 遍历整棵树,每个 dataset 的种类
    默认都是 "raw"、checked=False。使用者通过 describe() 浏览,用 set_kind(key, kind)
    显式标注语义(置 checked=True),再用 render(key) 渲染;或用 explore() 交互式选择。
    """

    def __init__(self, path: str):
        super().__init__(path)
        self.file = None  # 打开的 h5py.File 句柄(大数据惰性读取)
        self.index = None  # {key: {kind, checked, shape, dtype}} 全部 dataset 索引
        logger.info(
            "初始化 HDF5Reader | path=%s | file=%s | index=%s",
            path,
            self.file,
            self.index,
        )

    def load(self):
        """打开 HDF5,遍历所有 dataset 建立索引(种类默认 raw、未确认)。"""
        self.file = h5py.File(self.path, "r")
        self.index = {}
        self.file.visititems(self._on_item)
        logger.info("已加载 HDF5 %s | dataset 数=%d", self.path, len(self.index))
        return self.index

    def _on_item(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            # 不猜种类:一律默认 raw,由使用者 set_kind 确认
            self.index[name] = {
                "kind": "raw",
                "checked": False,
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }

    def describe(self):
        """打印全部 dataset:确认标记 / key / 当前种类 / shape / dtype,并返回索引。"""
        self._require_loaded()
        print(f"HDF5: {self.path}  ({len(self.index)} datasets)")
        for i, (key, e) in enumerate(self.index.items()):
            mark = "✓" if e["checked"] else "?"
            print(
                f"  [{i:>2}] {mark} {key:<45} kind={e['kind']:<7} "
                f"shape={e['shape']} dtype={e['dtype']}"
            )
        return self.index

    def set_kind(self, key, kind):
        """显式标注某个 key 的语义种类,并标记为已确认(checked=True)。"""
        self._require_loaded()
        if key not in self.index:
            raise KeyError(f"未知 key: {key}")
        if kind not in KINDS:
            raise ValueError(f"非法 kind {kind!r},可选: {KINDS}")
        self.index[key]["kind"] = kind
        self.index[key]["checked"] = True
        logger.info("标注 %s -> kind=%s (checked)", key, kind)
        return self.index[key]

    def render(self, key, frame=0):
        """按 key 已标注的种类渲染(未标注则为 raw 兜底)。种类用 set_kind 提前指定。"""
        self._require_loaded()
        if key not in self.index:
            raise KeyError(f"未知 key: {key}")
        kind = self.index[key]["kind"]
        dset = self.file[key]
        logger.info("渲染 %s | kind=%s | frame=%s", key, kind, frame)

        if kind == "image":
            self._render_image(dset, key, frame)
        elif kind == "series":
            self._render_series(dset, key)
        elif kind == "text":
            raw = dset[()] if dset.ndim == 0 else dset[frame]
            text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            try:  # 能解析成 JSON 就顺手美化,纯渲染细节
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except Exception:
                pass
            print(f"=== {key} (text) ===\n{text}")
        else:  # raw 兜底:不解释内容,只给结构和首元素预览
            preview = dset[()] if dset.ndim == 0 else np.asarray(dset[0]).ravel()[:16]
            print(f"=== {key} (raw) shape={dset.shape} dtype={dset.dtype} ===")
            print(f"首元素预览: {preview!r}")

    def _render_image(self, dset, key, frame):
        import matplotlib.pyplot as plt

        buf = dset[frame]
        img = cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("%s[%s] 无法解码为图像", key, frame)
            return
        plt.figure()
        if img.ndim == 3:  # 彩色 BGR -> RGB
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:  # 深度等单通道
            plt.imshow(img, cmap="viridis")
            plt.colorbar()
        plt.title(f"{key}  (frame {frame})")
        plt.axis("off")
        plt.show()

    def _render_series(self, dset, key):
        import matplotlib.pyplot as plt

        arr = dset[:]
        plt.figure()
        if arr.ndim == 1:
            plt.plot(arr)
        else:  # (T, N):每列一条曲线
            for c in range(arr.shape[1]):
                plt.plot(arr[:, c], label=f"[{c}]")
            plt.legend(fontsize="small", ncol=2)
        plt.title(key)
        plt.xlabel("frame")
        plt.tight_layout()
        plt.show()

    def explore(self):
        """交互式:列出所有 key,由使用者输入序号 + 可视化方式,选定即视为已确认。"""
        self._require_loaded()
        keys = list(self.index)
        while True:
            self.describe()
            sel = input("选择序号(回车退出)> ").strip()
            if not sel:
                break
            try:
                key = keys[int(sel)]
            except (ValueError, IndexError):
                print("无效序号,请重试")
                continue
            kind = input(f"可视化方式 {KINDS} [默认 raw]> ").strip() or "raw"
            if kind not in KINDS:
                print(f"非法种类,可选 {KINDS}")
                continue
            frame = 0
            if kind == "image":
                f = input("帧号 [默认 0]> ").strip()
                frame = int(f) if f else 0
            try:
                self.set_kind(key, kind)
                self.render(key, frame=frame)
            except Exception as exc:
                logger.warning("渲染失败: %s", exc)

    def close(self):
        """关闭底层 HDF5 文件句柄。"""
        if self.file is not None:
            self.file.close()
            self.file = None

    def _require_loaded(self):
        if self.file is None:
            self.load()
        return self.index
