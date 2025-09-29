from __future__ import annotations

import sys
from typing import Iterable, List, Optional

from .compute import collate
from .io_ndjson import write_rows
from .logging_cfg import setup_logging


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print("Usage: python -m core.cli URL_FILE", file=sys.stderr)
        return 1
    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows: Iterable[dict] = collate(line.strip() for line in f if line.strip())
            write_rows(rows)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
