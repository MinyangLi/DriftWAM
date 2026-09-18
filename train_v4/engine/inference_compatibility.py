"""Exact source-pair compatibility for unchanged cached teacher inference.

The training-mask repair changes model.py's file hash but preserves its torch
KV-cache inference path. Before registering this pair, FP32/BF16 tiny native
models were compared through 50 action steps and input-video gradients with
bitwise identical outputs. Full details are in the checked-in validation JSON.
Unknown source edits retain their own hash and still invalidate old caches.
"""

INFERENCE_COMPATIBLE_SOURCE_HASHES = {
    # Training-only eager mask construction; cached inference AST unchanged.
    '4b736fc986f09d6fa59b941ac6e64a5ffe04349d1bada641e0d1feda655fbfbb': 'b0e65d1f7df055f2b7d1b2d77b7de34b540e70414bb48f355ee3f43aabe4ff42',
    'f4c13b1a052fafec1e48ffbe236895b7de26e6102514faf3ef775cc7bd95c8a3': 'b0e65d1f7df055f2b7d1b2d77b7de34b540e70414bb48f355ee3f43aabe4ff42',
}


def inference_contract_source_hash(actual_source_hash: str) -> str:
    return INFERENCE_COMPATIBLE_SOURCE_HASHES.get(actual_source_hash, actual_source_hash)
