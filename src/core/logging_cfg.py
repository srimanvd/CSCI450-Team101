from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging() -> None:
    # 0 = silent (default), 1 = INFO, 2 = DEBUG
    level_map = {"0": logging.CRITICAL, "1": logging.INFO, "2": logging.DEBUG}
    lvl = level_map.get(os.getenv("LOG_LEVEL", "0"), logging.CRITICAL)

    log_file: Optional[str] = os.getenv("LOG_FILE")
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    if handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=handlers,
        )
    else:
        # Avoid attaching default stderr handler; keep stdout clean for NDJSON
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(lvl)
