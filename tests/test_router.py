import asyncio

import pytest

from liminallm.service.router import RouterEngine


def test_router_applies_rules_and_normalizes():
    engine = RouterEngine()
    policy = {
        "rules": [
            {"id": "activate_primary", "when": "true", "action": {"type": "activate_adapter_by_id", "adapter_id": "a1", "weight": 0.8}},
            {"id": "scale_secondary", "when": "true", "action": {"type": "activate_adapter_by_id", "adapter_id": "a2", "weight": 0.6}},
        ],
        "max_active_adapters": 2,
    }
    adapters = [
        {"id": "a1", "embedding": [1.0, 0.0]},
        {"id": "a2", "embedding": [0.0, 1.0]},
    ]
    ctx_emb = [1.0, 0.0]

    result = asyncio.run(engine.route(policy, ctx_emb, adapters))

    assert result["adapters"] == [
        {"id": "a1", "weight": pytest.approx(0.5714, rel=1e-3)},
        {"id": "a2", "weight": pytest.approx(0.4286, rel=1e-3)},
    ]
    assert result["trace"][0]["fired"] is True


def test_router_similarity_boost_when_no_rules_fire():
    engine = RouterEngine()
    adapters = [
        {"id": "similar", "embedding": [0.7, 0.7]},
        {"id": "dissimilar", "embedding": [0.0, 1.0]},
    ]
    ctx_emb = [0.7, 0.7]

    result = asyncio.run(engine.route({}, ctx_emb, adapters))

    assert any(entry["id"] == "default_similarity_boost" for entry in result["trace"])
    assert result["adapters"][0]["id"] == "similar"


def test_hash_embedding_preserves_precision():
    engine = RouterEngine()

    close = [0.12345, 0.98765]
    closer = [0.12346, 0.98765]

    assert engine._hash_embedding(close) != engine._hash_embedding(closer)
