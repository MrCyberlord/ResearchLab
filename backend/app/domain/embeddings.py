"""Stable identities of embedding spaces, independent of provider libraries."""
import hashlib
import json

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def profile_id(model: str, revision: str, pipeline: str, max_tokens: int | None = None) -> str:
    contract = {"model": model, "revision": revision, "pipeline": pipeline,
                "dimensions": 384, "normalization": "l2", "max_tokens": max_tokens}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


LEXICAL_PROFILE = profile_id("local-hash-v1", "local-hash-v1", "nfkc-casefold-sha256-signed-v1")
