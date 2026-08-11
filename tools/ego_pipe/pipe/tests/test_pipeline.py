from dataclasses import dataclass, field

import pytest

from tools.ego_pipe.pipe import Pipeline


def test_empty_pipeline_returns_input_unchanged() -> None:
    pipeline: Pipeline = Pipeline([])

    result = pipeline.run(42)

    assert result == 42


@dataclass
class _Counter:
    value: int = 0
    log: list[str] = field(default_factory=list)


def test_steps_mutate_same_object_in_place() -> None:
    def increment(c: _Counter) -> None:
        c.value += 1
        c.log.append("increment")

    def double(c: _Counter) -> None:
        c.value *= 2
        c.log.append("double")

    pipeline: Pipeline = Pipeline([increment, double])
    obj = _Counter(value=1)

    result = pipeline.run(obj)

    assert result is obj  # 同一个对象,身份不变
    assert result.value == 4  # (1 + 1) * 2
    assert result.log == ["increment", "double"]


def test_step_failure_is_wrapped_with_index_and_name_and_keeps_cause() -> None:
    def ok(c: _Counter) -> None:
        c.value += 1

    def boom(c: _Counter) -> None:
        raise ValueError("bad value")

    pipeline: Pipeline = Pipeline([ok, boom])

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run(_Counter())

    message = str(exc_info.value)
    assert "1" in message  # step index
    assert "boom" in message  # step name
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_run_logs_step_list_and_progress(caplog: pytest.LogCaptureFixture) -> None:
    def increment(c: _Counter) -> None:
        c.value += 1

    def double(c: _Counter) -> None:
        c.value *= 2

    pipeline: Pipeline = Pipeline([increment, double])

    with caplog.at_level("INFO"):
        pipeline.run(_Counter())

    assert "2" in caplog.text  # steps 总数
    assert "increment" in caplog.text
    assert "double" in caplog.text
    assert "1/2" in caplog.text  # 第一个 step 的进度标记
    assert "2/2" in caplog.text  # 第二个 step 的进度标记
