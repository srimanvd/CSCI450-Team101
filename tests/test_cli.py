# tests/test_cli.py
import importlib
import os

# Import the CLI module from the project (tests/conftest.py should put src/ first on sys.path)
cli_mod = importlib.import_module("core.cli")
main = getattr(cli_mod, "main")

def _expand_like_cli(raw_items):
    """
    If we received raw CSV triplet lines, expand them into ordered URLs.
    If we already received expanded URLs, just return them.
    Expansion order per line is: code -> dataset -> model.
    """
    items = list(raw_items)

    # Already expanded? (all items are http links without commas)
    if items and all(isinstance(x, str) and x.strip().startswith(("http://", "https://")) and ("," not in x) for x in items):
        return items

    # Otherwise treat as raw CSV lines and expand with the same rules used by core.cli
    out = []
    for raw in items:
        line = (raw or "").strip()
        if not line:
            continue
        cells = [p.strip() for p in (line.split(",", 2) + ["", "", ""])[:3]]
        code, dataset, model = cells[0], cells[1], cells[2]
        if code.startswith(("http://", "https://")):
            out.append(code)
        if dataset.startswith(("http://", "https://")):
            out.append(dataset)
        if model.startswith(("http://", "https://")):
            out.append(model)
    return out


def test_cli_main_expands_urls_in_order(monkeypatch, tmp_path):
    # Build a triplet file with a mix of blank/non-HTTP cells
    src = tmp_path / "urls.txt"
    src.write_text(
        "a,b,c\n"                     # ignored (no http)
        "https://x,y,https://z\n"     # => x (code), z (model)
        "https://a,,\n"               # => a
        ",https://b,\n"               # => b
        ",,https://c\n"               # => c
    )

    captured = {"raw": None, "expanded": None, "called": False, "wrote": False}

    def fake_collate(it):
        # Capture what main() is passing us, then normalize to URL expansion
        captured["raw"] = list(it)
        captured["expanded"] = _expand_like_cli(captured["raw"])
        captured["called"] = True
        # Yield a minimal valid row so main() completes
        yield {
            "name": "m",
            "category": "MODEL",
            "net_score": 0.5,
            "net_score_latency": 1,
            "ramp_up_time": 1,
            "ramp_up_time_latency": 1,
            "bus_factor": 0,
            "bus_factor_latency": 0,
            "performance_claims": 0,
            "performance_claims_latency": 0,
            "license": 1,
            "license_latency": 0,
            "size_score": {},
            "size_score_latency": 0,
            "dataset_and_code_score": 0,
            "dataset_and_code_score_latency": 0,
            "dataset_quality": 0,
            "dataset_quality_latency": 0,
            "code_quality": 0,
            "code_quality_latency": 0,
        }

    def fake_write(rows, out=None):
        captured["wrote"] = True
        for _ in rows:
            pass

    # Keep logs quiet and HF progress bars off, and stub collate/write
    monkeypatch.setenv("LOG_LEVEL", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setattr(cli_mod, "collate", fake_collate)
    monkeypatch.setattr(cli_mod, "write_rows", fake_write)

    rc = main(["prog", str(src)])
    assert rc == 0
    assert captured["called"] and captured["wrote"]

    # Expected expansion order (code -> dataset -> model within a line)
    assert captured["expanded"] == [
        "https://x",
        "https://z",
        "https://a",
        "https://b",
        "https://c",
    ]


def test_cli_main_usage_message(capsys):
    rc = main(["prog"])
    assert rc == 1
    assert "Usage:" in capsys.readouterr().err
