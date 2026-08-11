"""ego_pipe 全模块共用的标准日志格式。只在入口脚本(example/CLI)里调用,库模块本身不配置 logging。"""

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_FORMAT)
