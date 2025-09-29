from __future__ import annotations

import sys
from typing import Iterable

from .compute import collate
from .io_ndjson import write_rows
from .logging_cfg import setup_logging


def _iter_urls_from_triplet_file(path: str) -> Iterable[str]:
    """
    Each non-empty line must be a CSV triplet:
        <code_link>, <dataset_link>, <model_link>
    We expand each line into up to 3 URLs in order: code -> dataset -> model.
    Empty/missing cells are skipped. There is NO cross-line inheritance.
    """
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # split into at most 3 cells; trim whitespace
            c = [p.strip() for p in (line.split(",", 2) + ["", "", ""])[:3]]
            code, dataset, model = c[0], c[1], c[2]
            if code.startswith(("http://", "https://")):
                yield code
            if dataset.startswith(("http://", "https://")):
                yield dataset
            if model.startswith(("http://", "https://")):
                yield model


def main(argv=None) -> int:
    setup_logging()
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print("Usage: python -m core.cli URL_FILE", file=sys.stderr)
        return 1
    path = argv[1]
    try:
        urls = _iter_urls_from_triplet_file(path)
        rows = collate(urls)  # collate already associates preceding code/dataset with the next model
        write_rows(rows)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
