"""reasoning_effort passthrough for OpenAI-compatible reasoning models."""
from types import SimpleNamespace

from liminallm.service.model_backend import ApiAdapterBackend


class _CaptureClient:
    """Fake OpenAI client that records the kwargs of the completion call."""

    def __init__(self):
        self.kwargs = None
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(choices=[choice], usage=usage)


def _backend(**kwargs) -> tuple[ApiAdapterBackend, _CaptureClient]:
    backend = ApiAdapterBackend(
        "gemini-flash-latest", api_key="test-key", provider="gemini", **kwargs
    )
    client = _CaptureClient()
    backend.client = client
    backend._ensure_client = lambda: None  # keep the fake client installed
    return backend, client


def test_reasoning_effort_sent_in_extra_body():
    backend, client = _backend(reasoning_effort="low")
    backend.generate([{"role": "user", "content": "hi"}], [])
    assert client.kwargs["extra_body"]["reasoning_effort"] == "low"


def test_reasoning_effort_omitted_when_unset():
    backend, client = _backend()
    backend.generate([{"role": "user", "content": "hi"}], [])
    extra = client.kwargs["extra_body"]
    assert not extra or "reasoning_effort" not in extra


def test_reasoning_effort_is_normalised():
    backend, client = _backend(reasoning_effort="  High  ")
    backend.generate([{"role": "user", "content": "hi"}], [])
    assert client.kwargs["extra_body"]["reasoning_effort"] == "high"


def test_the_backend_does_not_read_the_environment(monkeypatch):
    """It used to. reasoning_effort is an admin-managed setting now, so it
    arrives as an argument; a backend reaching into os.environ would put it
    back to being configurable per container and invisible to the admin UI."""
    monkeypatch.setenv("MODEL_REASONING_EFFORT", "high")
    backend, client = _backend()
    backend.generate([{"role": "user", "content": "hi"}], [])
    extra = client.kwargs["extra_body"]
    assert not extra or "reasoning_effort" not in extra


def test_the_setting_reaches_the_backend_from_settings():
    """The whole path: managed setting -> LLMService -> backend."""
    from liminallm.config import Settings, apply_managed_settings
    from liminallm.service.llm import LLMService

    settings = apply_managed_settings(
        Settings(), {"model_reasoning_effort": "medium"}
    )
    llm = LLMService(
        base_model="gemini-flash-latest",
        api_key="test-key",
        reasoning_effort=settings.model_reasoning_effort,
    )
    assert llm.backend._reasoning_effort == "medium"
