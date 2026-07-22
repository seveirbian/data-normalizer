import logging
from abc import ABC, abstractmethod

import yourdfpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)


class RobotReader(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load(self):
        """读取描述文件,解析成内部数据结构"""
        ...

    @abstractmethod
    def validate(self):
        """校验描述文件自身是否合法、自洽"""
        ...

    @abstractmethod
    def visualize(self):
        """可视化机器人结构/姿态"""
        ...

class URDFReader(RobotReader):
    """基于 yourdfpy 的 URDF 描述文件读取器。"""

    def __init__(self, path: str, load_meshes: bool = True):
        super().__init__(path)
        self.load_meshes = load_meshes
        self.robot = None
        logger.info(
            "初始化 URDFReader | path=%s | load_meshes=%s | robot=%s",
            path,
            load_meshes,
            self.robot,
        )

    def load(self):
        """读取 URDF,解析成 yourdfpy.URDF 对象。"""
        self.robot = yourdfpy.URDF.load(self.path, load_meshes=self.load_meshes)
        logger.info(
            "已加载 URDF %s | joint数=%d(可驱动 %d) | link数=%d",
            self.path,
            len(self.robot.joint_map),
            self.robot.num_actuated_joints,
            len(self.robot.link_map),
        )
        return self.robot

    def validate(self):
        """校验 URDF 是否合法、自洽,返回 bool;错误细节见 self.robot.errors。"""
        robot = self._require_loaded()
        robot.clear_errors()
        ok = robot.validate()
        if ok:
            logger.info("URDF 校验通过: %s", self.path)
        else:
            logger.warning("URDF 校验未通过: %s | 错误: %s", self.path, robot.errors)
        return ok

    def visualize(self):
        """在交互式窗口中可视化机器人结构/姿态(默认零位)。"""
        self._require_loaded().show()

    def _require_loaded(self):
        if self.robot is None:
            self.load()
        return self.robot