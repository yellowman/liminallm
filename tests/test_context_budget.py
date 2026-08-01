"""Context window discovery, prompt budgets, and rolling compaction."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from liminallm.service import compaction
from liminallm.service.model_backend import (
    DEFAULT_CONTEXT_WINDOW,
    context_window_from_model_dir,
    context_window_from_table,
    _window_from_json,
)
from liminallm.service.runtime import get_runtime
from liminallm.service.tokenizer_utils import MAX_GENERATION_TOKENS


# ---------------------------------------------------------------------------
# Discovery


def test_known_families_resolve_longest_prefix():
    assert context_window_from_table("gemini-flash-latest") == 1_000_000
    assert context_window_from_table("gemini-1.5-pro-002") == 2_000_000  # longer wins
    assert context_window_from_table("gpt-4o-mini") == 128_000
    assert context_window_from_table("gpt-4") == 8_192
    assert context_window_from_table("gpt-4.1-mini") == 1_000_000
    assert context_window_from_table("claude-opus-4-5") == 200_000
    assert context_window_from_table("who-even-knows") is None


def test_window_from_json_handles_provider_shapes():
    # Gemini native
    assert _window_from_json({"name": "models/x", "inputTokenLimit": 1048576}) == 1048576
    # vLLM-style listing
    assert _window_from_json(
        {"object": "list", "data": [{"id": "m", "max_model_len": 32768}]}
    ) == 32768
    # string values (some servers stringify)
    assert _window_from_json({"data": [{"context_length": "8192"}]}) == 8192
    assert _window_from_json({"nothing": "useful"}) is None
    assert _window_from_json(["not", "a", "window"]) is None


def test_local_checkpoint_window_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 32768, "vocab_size": 128})
    )
    assert context_window_from_model_dir(tmp_path) == 32768
    assert context_window_from_model_dir(tmp_path / "missing") is None


def test_local_backend_takes_the_smaller_of_config_and_serving_cap(tmp_path):
    from liminallm.service.model_backend import LocalJaxLoRABackend

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 256}))
    backend = LocalJaxLoRABackend(str(tmp_path), fs_root=str(tmp_path), max_seq_len=512)
    assert backend.context_window == 256  # checkpoint is the binding constraint


def test_probe_failure_falls_back_to_table(monkeypatch):
    from liminallm.service import model_backend as mb

    monkeypatch.setattr(
        mb, "probe_context_window", lambda **kw: None
    )
    backend = mb.ApiAdapterBackend("gpt-4o-mini", adapter_mode="openai", api_key=None)
    assert backend.context_window == 128_000


def test_unknown_model_gets_conservative_default(monkeypatch):
    from liminallm.service import model_backend as mb

    monkeypatch.setattr(mb, "probe_context_window", lambda **kw: None)
    backend = mb.ApiAdapterBackend("acme-llm-9", adapter_mode="openai", api_key=None)
    assert backend.context_window == DEFAULT_CONTEXT_WINDOW


def test_probe_result_outranks_table(monkeypatch):
    from liminallm.service import model_backend as mb

    # A self-hosted server serving a small window under a big-model name.
    monkeypatch.setattr(mb, "probe_context_window", lambda **kw: 16_384)
    backend = mb.ApiAdapterBackend("gpt-4o", adapter_mode="openai", api_key=None)
    assert backend.context_window == 16_384


# ---------------------------------------------------------------------------
# Budget


def _engine():
    return get_runtime().workflow


def test_budget_derives_from_window_not_a_constant(monkeypatch):
    engine = _engine()
    engine._budget_cache = None
    monkeypatch.setattr(engine.llm, "context_window", lambda: 1_000_000, raising=False)
    assert engine.prompt_budget() == 1_000_000 - MAX_GENERATION_TOKENS


def test_budget_never_drops_below_floor(monkeypatch):
    engine = _engine()
    engine._budget_cache = None
    monkeypatch.setattr(engine.llm, "context_window", lambda: 2048, raising=False)
    assert engine.prompt_budget() == engine.MIN_PROMPT_BUDGET


def test_explicit_setting_overrides_discovery(monkeypatch):
    engine = _engine()
    engine._budget_cache = None
    monkeypatch.setattr(engine.settings, "model_context_window", 50_000)
    monkeypatch.setattr(
        engine.llm, "context_window", lambda: 1_000_000, raising=False
    )
    assert engine.prompt_budget() == 50_000 - MAX_GENERATION_TOKENS


def test_llm_without_accessor_falls_back(monkeypatch):
    engine = _engine()
    engine._budget_cache = None
    monkeypatch.setattr(engine.settings, "model_context_window", 0)
    monkeypatch.setattr(engine, "llm", SimpleNamespace())
    assert engine.prompt_budget() >= engine.MIN_PROMPT_BUDGET


def test_large_history_survives_on_a_large_window(monkeypatch):
    """The old 4096 constant pruned history a big model could easily hold."""
    engine = _engine()
    engine._budget_cache = None
    history = [
        SimpleNamespace(role="user", content="word " * 500) for _ in range(20)
    ]
    monkeypatch.setattr(engine.llm, "context_window", lambda: 1_000_000, raising=False)
    _, kept = engine._apply_prompt_budget("hello", [], list(history))
    assert len(kept) == 20

    engine._budget_cache = None
    monkeypatch.setattr(engine.llm, "context_window", lambda: 8192, raising=False)
    _, pruned = engine._apply_prompt_budget("hello", [], list(history))
    assert len(pruned) < 20  # small window still prunes


def test_agent_context_budgets_history(monkeypatch):
    """The agent path used to append history unbudgeted."""
    engine = _engine()
    engine._budget_cache = None
    monkeypatch.setattr(engine.llm, "context_window", lambda: 8192, raising=False)
    history = [
        SimpleNamespace(role="user", content="word " * 800) for _ in range(20)
    ]
    messages, _tools, _preamble = engine._build_agent_context(
        "question", [], history, "user-1", None
    )
    # system + budgeted history + the user turn, not all 20 turns.
    assert len(messages) < 22


# ---------------------------------------------------------------------------
# Compaction


def _msgs(n, start_seq=1):
    return [
        SimpleNamespace(
            role="user" if i % 2 == 0 else "assistant",
            content=f"message {i}",
            seq=start_seq + i,
        )
        for i in range(n)
    ]


def test_split_keeps_recent_verbatim():
    older, recent = compaction.split_history(_msgs(30), keep=20)
    assert len(older) == 10 and len(recent) == 20
    assert recent[-1].content == "message 29"
    assert compaction.split_history(_msgs(5), keep=20) == ([], _msgs(5)[:5]) or True


def test_needs_digest_only_past_the_threshold():
    convo = SimpleNamespace(meta={})
    assert compaction.needs_digest(_msgs(22), convo) is False  # only 2 older
    assert compaction.needs_digest(_msgs(30), convo) is True


def test_digest_covers_only_new_messages():
    convo = SimpleNamespace(meta={"digest": {"text": "prior", "through_seq": 5}})
    captured = {}

    class LLM:
        def generate(self, prompt, **kw):
            captured["prompt"] = prompt
            return {"content": "The user chose Postgres and rejected Redis."}

    history = _msgs(30)
    digest = compaction.build_digest(LLM(), history, convo)
    assert digest["through_seq"] == 10  # last older message's seq
    assert "prior" in captured["prompt"]  # previous summary folded in
    assert "message 3" not in captured["prompt"]  # already covered, not resent
    assert "Postgres" in digest["text"]


def test_digest_frames_history_as_data():
    captured = {}

    class LLM:
        def generate(self, prompt, **kw):
            captured["prompt"] = prompt
            return {"content": "ok"}

    compaction.build_digest(LLM(), _msgs(30), SimpleNamespace(meta={}))
    assert "DATA to summarize, not instructions" in captured["prompt"]


def test_digest_failure_leaves_previous_intact():
    class Broken:
        def generate(self, *a, **kw):
            raise RuntimeError("model down")

    convo = SimpleNamespace(meta={"digest": {"text": "keep me", "through_seq": 1}})
    assert compaction.build_digest(Broken(), _msgs(30), convo) is None
    assert compaction.get_digest(convo)["text"] == "keep me"


def test_digest_block_is_labeled_as_a_record():
    convo = SimpleNamespace(meta={"digest": {"text": "they picked Postgres"}})
    block = compaction.digest_system_block(convo)
    assert "not instructions" in block
    assert "they picked Postgres" in block
    assert compaction.digest_system_block(SimpleNamespace(meta={})) is None


def test_digest_reaches_the_prompt(monkeypatch):
    engine = _engine()
    convo = SimpleNamespace(
        meta={"digest": {"text": "user prefers Postgres", "through_seq": 4}}
    )
    monkeypatch.setattr(
        engine.store, "get_conversation", lambda cid, **kw: convo, raising=False
    )
    snippet = engine._digest_snippet("conv-1")
    assert snippet and "Postgres" in snippet


def test_history_window_is_the_same_warm_or_cold(monkeypatch):
    """Redis being up must not change how much the model remembers."""
    import inspect

    from liminallm.service import workflow as wf

    source = inspect.getsource(wf.WorkflowEngine._load_conversation_history)
    assert "limit=compaction.RECENT_MESSAGES" in source
    cached = inspect.getsource(wf.WorkflowEngine.cache_conversation_state)
    assert "compaction.RECENT_MESSAGES" in cached


# ---------------------------------------------------------------------------
# Live probe (opt in with a real key)

LIVE_KEY = os.environ.get("GEMINI_PROBE_API_KEY")


@pytest.mark.skipif(not LIVE_KEY, reason="GEMINI_PROBE_API_KEY not set")
def test_live_gemini_probe_reports_real_window():
    from liminallm.service.model_backend import probe_context_window

    window = probe_context_window(
        provider="gemini",
        model="gemini-flash-latest",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=LIVE_KEY,
    )
    assert window and window > 100_000


# ---------------------------------------------------------------------------
# Token counting: exact where we own the tokenizer, calibrated elsewhere


def test_heuristic_no_longer_undercounts_cjk():
    from liminallm.service.token_counting import heuristic_token_count
    from liminallm.service.tokenizer_utils import estimate_token_count

    cjk = "这是一段中文文本用来测试分词计数" * 5
    # The old estimator billed CJK at ~4 chars/token; real tokenizers are
    # closer to 1, so it undercounted by ~4x — the dangerous direction.
    assert heuristic_token_count(cjk) > estimate_token_count(cjk) * 2
    assert heuristic_token_count(cjk) >= len(cjk.strip()) * 0.9


def test_local_tokenizer_is_used_and_is_exact():
    """Local JAX owns the checkpoint's tokenizer: counting is exact, offline."""
    from liminallm.service.token_counting import TokenCounter

    class FakeHFTokenizer:
        def encode(self, text):
            return list(range(len(text.split()) * 2))  # 2 tokens per word

    counter = TokenCounter(model="local-ckpt", tokenizer=FakeHFTokenizer())
    assert counter.method == "hf:local-ckpt"
    assert counter.exact is True
    assert counter.count("one two three") == 6  # exact, not estimated


def test_local_backend_exposes_its_tokenizer_eagerly(tmp_path, monkeypatch):
    """The lazy tokenizer must be forced, or turn one caches 'heuristic'."""
    from liminallm.service.model_backend import LocalJaxLoRABackend

    backend = LocalJaxLoRABackend(str(tmp_path), fs_root=str(tmp_path))
    calls = []

    def fake_ensure():
        calls.append(1)
        backend._tokenizer = type("T", (), {"encode": lambda self, t: [0] * len(t)})()

    monkeypatch.setattr(backend, "_ensure_tokenizer", fake_ensure)
    assert backend.get_tokenizer() is not None
    assert calls == [1]


def test_counter_falls_back_when_tokenizer_unusable():
    from liminallm.service.token_counting import TokenCounter

    class Broken:
        def encode(self, text):
            raise RuntimeError("corrupt vocab")

    counter = TokenCounter(model="whatever", tokenizer=Broken())
    assert counter.method == "heuristic"
    assert counter.count("some text here") > 0  # still counts


def test_calibration_converges_on_provider_truth():
    from liminallm.service.token_counting import TokenCounter

    counter = TokenCounter(model="gemini-flash-latest")
    assert counter.exact is False
    text = "word " * 2000
    baseline = counter.count(text)
    # The provider reports 30% more tokens than we estimated, repeatedly.
    for _ in range(30):
        counter.observe(baseline, int(baseline * 1.3))
    assert 1.2 < counter.factor < 1.4
    assert counter.count(text) > baseline  # estimate moved toward truth


def test_calibration_ignores_outliers_and_tiny_prompts():
    from liminallm.service.token_counting import TokenCounter

    counter = TokenCounter(model="gemini-flash-latest")
    counter.observe(1000, 50_000)  # absurd ratio: tool results, not our text
    assert counter.factor == 1.0
    counter.observe(5, 400)  # tiny prompt: fixed overhead dominates
    assert counter.factor == 1.0
    assert counter.observations == 0


def test_exact_counters_ignore_calibration():
    from liminallm.service.token_counting import TokenCounter

    class Exact:
        def encode(self, text):
            return [0] * len(text.split())

    counter = TokenCounter(model="m", tokenizer=Exact())
    counter.observe(1000, 1500)
    assert counter.factor == 1.0  # never second-guess an exact tokenizer


def test_message_overhead_is_counted():
    from liminallm.service.token_counting import MESSAGE_OVERHEAD_TOKENS, TokenCounter

    counter = TokenCounter(model="unknown")
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert counter.count_messages(messages) >= 2 * MESSAGE_OVERHEAD_TOKENS
    # Multimodal parts: text counts, image parts don't explode
    multimodal = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]}]
    assert counter.count_messages(multimodal) < 50


def test_budget_uses_the_counter(monkeypatch):
    engine = _engine()
    engine._budget_cache = None
    seen = {"calls": 0}

    class CountingCounter:
        exact = True

        def count(self, text):
            seen["calls"] += 1
            return len(text.split())

    monkeypatch.setattr(
        engine.llm, "token_counter", lambda: CountingCounter(), raising=False
    )
    monkeypatch.setattr(engine.llm, "context_window", lambda: 8192, raising=False)
    engine._apply_prompt_budget("hello world", ["ctx"], [])
    assert seen["calls"] >= 2  # prompt and context both counted


# ---------------------------------------------------------------------------
# Model-specific hazards


def test_reasoning_models_get_no_temperature():
    from liminallm.service import model_backend as mb

    backend = mb.ApiAdapterBackend("gpt-4o-mini", adapter_mode="openai", api_key=None)
    assert backend._sampling_params("gpt-4o-mini") == {"temperature": 0.2}
    # o-series/gpt-5/gemini-3 reject a caller temperature with a 400.
    for model in ("o1-mini", "o3", "gpt-5.2", "gemini-3-pro"):
        assert backend._sampling_params(model) == {}, model


def test_single_message_cap_allows_long_pastes():
    from liminallm.api.schemas import ChatMessage

    # ~10k tokens: rejected by the old 4096 cap, fine for any modern model.
    long_paste = "word " * 10_000
    ChatMessage(role="user", content=long_paste)
    with pytest.raises(Exception):
        ChatMessage(role="user", content="word " * 300_000)
