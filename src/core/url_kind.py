from __future__ import annotations
from dataclasses import dataclass
from urllib.parse import urlparse

@dataclass(frozen=True)
class ParsedURL:
    raw: str
    kind: str
    owner: str | None
    name: str | None

def parse_url(u: str) -> ParsedURL:
    p = urlparse(u.strip())
    host = p.netloc.lower()

    if host in {"huggingface.co", "www.huggingface.co"}:
        parts = [x for x in p.path.split("/") if x]
        # canonical: /{owner}/{repo}
        if len(parts) >= 2:
            return ParsedURL(u, "hf_model", parts[0], parts[1])
        return ParsedURL(u, "other", None, None)

    if host in {"github.com", "www.github.com"}:
        parts = [x for x in p.path.split("/") if x]
        if len(parts) >= 2:
            return ParsedURL(u, "github", parts[0], parts[1])
        return ParsedURL(u, "other", None, None)

    return ParsedURL(u, "other", None, None)
