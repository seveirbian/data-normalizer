import logging

from tools.ego_pipe.log import configure_logging


def test_configure_logging_sets_info_level_and_standard_format() -> None:
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()

    try:
        configure_logging()

        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        fmt = root.handlers[0].formatter._fmt
        assert fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    finally:
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)
