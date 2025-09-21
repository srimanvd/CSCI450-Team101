from core.url_kind import parse_url
from core.metrics import compute_metrics

def test_parse_hf_model():
    p = parse_url("https://huggingface.co/facebook/opt-125m")
    assert p.kind == "hf_model"
    assert p.owner == "facebook"
    assert p.name == "opt-125m"

def test_non_model_is_skipped():
    r = compute_metrics("https://github.com/pallets/flask")
    assert r == {}
