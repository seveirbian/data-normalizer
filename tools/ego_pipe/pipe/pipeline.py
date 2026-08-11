"""极简线性 pipeline 引擎:一串 callable 依次就地修改同一个对象,不携带任何领域语义。"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _step_name(step: Callable[[Any], None]) -> str:
    return getattr(step, "__name__", repr(step))


class Pipeline:
    """按顺序执行一串 step;每个 step 就地修改传入的同一个对象,不返回值。"""

    def __init__(self, steps: list[Callable[[Any], None]]) -> None:
        self.steps = steps

    def run(self, obj: Any) -> Any:
        n = len(self.steps)
        names = [_step_name(step) for step in self.steps]
        logger.info("%d steps -> %s", n, names)

        for i, step in enumerate(self.steps):
            name = names[i]
            logger.info("[%d/%d] running %s", i + 1, n, name)
            try:
                step(obj)
            except Exception as e:
                raise RuntimeError(f"pipeline step {i} ({name}) failed") from e
        return obj
