"""A provider's own replayable continuation, kept beside the readable record.

The trusted transcript is chat-shaped on purpose: every provider can be given
a conversation rebuilt from it, and the parent can say exactly what the worker
continued from. What it cannot carry is what a provider keeps for itself
between turns - the encrypted reasoning a Responses model expects returned,
the signed parts a Gemini candidate arrived with, the item types neither has
invented yet. Rebuilding a turn from its visible text and calls discards that,
and the model does its thinking over on every round.

So there are two records of one conversation, and neither is derived from the
other. The transcript remains the security truth: what was offered, what was
asked, what ran. This is the performance truth: what the provider said, in
the provider's own terms, so it can be handed back as the provider wants it.
It is opaque to everything except the adapter that wrote it, it lives on the
invocation context and nowhere shared, and it advances only when the parent
accepts a complete model turn - so a rejected turn, a divergent round, or a
dead worker leaves it exactly where the last accepted turn put it.

Zero encryption of our own. A provider's `encrypted_content`, signatures and
opaque blobs are ordinary values here, preserved as the JSON they arrived as
and replayed as required. Nothing wraps, re-encodes, or interprets them.

Every backend declares which kind of continuation it keeps - the two native
kinds, the chat-shaped one, or none - and the declaration is per mode, by
name. It is never inferred from a transport: a gateway that answers
`/responses` has OpenAI's wire, not OpenAI's entitlement to encrypted
reasoning, and `_infer_provider`'s "openai" fallback for an unknown mode is
precisely the default that must not decide this.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from liminallm.config import PROVIDER_ENDPOINTS

#: The strategies a backend may declare. Versioned in the name, because the
#: payload each one keeps is a wire format the provider may revise: a stored
#: `openai.responses.native.v1` state is one a `v2` adapter must recognise
#: as foreign rather than replay.
#:
#: The full ordered `response.output` of every accepted turn, replayed as
#: input with `store=false`. Encrypted reasoning rides inside it.
OPENAI_RESPONSES_NATIVE_V1 = "openai.responses.native.v1"
#: The complete selected candidate of every accepted turn, parts and
#: signatures in order, replayed as `contents`.
GEMINI_NATIVE_V1 = "gemini.native.v1"
#: The chat-shaped assistant message the transport handed back, kept whole
#: on the transcript entry so vendor extras survive the round trip. What
#: every OpenAI-compatible endpoint gets, whichever wire it happens to speak.
CHAT_STRUCTURED_V1 = "chat.structured.v1"
#: Nothing beyond the transcript. Local serving and the stub keep no state a
#: reconstruction would lose.
TRANSCRIPT_V1 = "transcript.v1"

STRATEGIES = frozenset({
    OPENAI_RESPONSES_NATIVE_V1,
    GEMINI_NATIVE_V1,
    CHAT_STRUCTURED_V1,
    TRANSCRIPT_V1,
})

#: The strategies that claim intact native continuation. A claim, so it is
#: made by exactly the adapters that keep it, and by no fallback.
NATIVE_STRATEGIES = frozenset({OPENAI_RESPONSES_NATIVE_V1, GEMINI_NATIVE_V1})

#: Modes with no API behind them.
_TRANSCRIPT_MODES = frozenset({"stub", "local_lora", "local_gpu_lora"})

#: OpenAI-compatible modes that resolve to an endpoint outside
#: `PROVIDER_ENDPOINTS` - a deployment-supplied URL, or a hosted adapter
#: server. The same wire and the same declaration as the table's rows.
#: `api_adapters` is the service's own default when it is built without a
#: mode: a compatible endpoint named by URL, with no provider behind the name.
_COMPAT_MODES_WITHOUT_ENDPOINT = frozenset({
    "api_adapters", "vertex", "bedrock", "lorax", "adapter_server",
    "sagemaker", "aws_sagemaker",
})


def declared_strategy(mode: str) -> str:
    """The continuation a backend mode declares, before any negotiation.

    Keyed on the literal mode. `openai` alone declares the Responses-native
    strategy: Azure serves the same models, but whether its Responses surface
    returns encrypted reasoning under `store=false` has not been verified
    here, and a weaker declaration is never wrong where a stronger one can
    be. Promote it when that is measured.

    Everything in `PROVIDER_ENDPOINTS` is an OpenAI-compatible provider, so
    the table is the census for the chat-shaped strategy and a new row gets
    it without a second registration. What is not in the table and not named
    below is refused: "openai" is what `_infer_provider` answers for a mode
    it does not know, and this is the one place that answer must never
    reach.
    """
    lowered = (mode or "").strip().lower()
    if lowered == "openai":
        return OPENAI_RESPONSES_NATIVE_V1
    if lowered == "gemini_native":
        return GEMINI_NATIVE_V1
    if lowered in _TRANSCRIPT_MODES:
        return TRANSCRIPT_V1
    if lowered in PROVIDER_ENDPOINTS or lowered in _COMPAT_MODES_WITHOUT_ENDPOINT:
        return CHAT_STRUCTURED_V1
    raise ValueError(f"backend mode {mode!r} declares no continuation strategy")


class ContinuationMismatch(RuntimeError):
    """What is serving this attempt is not on the continuation accepted.

    A different strategy, wire, provider or model than the record was
    written by, or a wire that cannot carry it at all. Raised by an adapter
    before it asks the provider, and by the parent when what came back is
    not on the record. Refused rather than adapted: continuing a provider's
    own state through something else is the substitution this record exists
    to make impossible, and a change needs an explicit reset, not a quiet
    one.
    """


class ModelTurnRejected(RuntimeError):
    """The model's reply is not a turn the parent can accept whole.

    All of it or none of it. A reply the provider itself reports as cut
    off, failed, cancelled or still running; a reply with nothing in it; a
    reply with one call whose arguments are not a JSON object - none of
    these is a reply with the rest of it in it. A call that looks whole
    inside a reply that was cut off is part of a reply that was cut off.
    Raised by an adapter for what only its wire can say, and by the parent
    for the shape it reads. Nothing of it is recorded, nothing in it runs,
    and the provider's continuation stays where the last accepted turn left
    it.
    """


@dataclass(frozen=True)
class ProviderContinuation:
    """One provider's accepted continuation, through one operation.

    `payload` is the provider's, in the provider's terms, and nothing here
    reads it. The common layer knows the lifecycle - which strategy wrote it,
    for which provider and model over which transport, and how far along the
    trusted transcript it reaches - and hands the payload back to the adapter
    that can use it.

    Deeply copied in and out, for the reason `ModelTurn` is: the ledger keeps
    this record for every later attempt, and a caller or reader editing a
    nested list in place would be editing what the next attempt is restored
    to.
    """

    strategy: str
    provider: str
    transport: str
    #: The model the state was produced against. Opaque reasoning is not
    #: portable across families, so a restore under another model is either
    #: a fresh state or a refusal, never a replay.
    model: str
    #: The last transcript operation this state accounts for. What the parent
    #: supplies on the next call is the entries after it - by identity, never
    #: by diffing text, because the two records deliberately differ in what
    #: they say a turn contained.
    through_operation_seq: int
    payload: Mapping[str, Any]
    #: What the state costs on the next request beyond the rendered
    #: transcript: the provider-reported reasoning tokens of every accepted
    #: turn, summed. The parent's accounting, not the adapter's, and reserved
    #: from the prompt budget when offers are priced - the transcript rebuild
    #: is what gets measured, and it carries none of this.
    replay_tokens: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"{self.strategy!r} is not a continuation strategy this "
                "release knows"
            )
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))
        object.__setattr__(
            self, "through_operation_seq", int(self.through_operation_seq)
        )
        object.__setattr__(self, "replay_tokens", max(0, int(self.replay_tokens or 0)))

    def __repr__(self) -> str:
        # Identity and size, never the payload: a repr is one log line away
        # from a leak, and this one has nothing to leak.
        sizes = {k: len(v) for k, v in self.payload.items() if isinstance(v, list)}
        return (
            f"ProviderContinuation(strategy={self.strategy!r}, "
            f"provider={self.provider!r}, transport={self.transport!r}, "
            f"model={self.model!r}, through_operation_seq="
            f"{self.through_operation_seq}, replay_tokens={self.replay_tokens}, "
            f"payload_sizes={sizes!r})"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "provider": self.provider,
            "transport": self.transport,
            "model": self.model,
            "through_operation_seq": self.through_operation_seq,
            "payload": deepcopy(dict(self.payload)),
            "replay_tokens": self.replay_tokens,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProviderContinuation":
        return cls(
            strategy=str(raw.get("strategy") or ""),
            provider=str(raw.get("provider") or ""),
            transport=str(raw.get("transport") or ""),
            model=str(raw.get("model") or ""),
            through_operation_seq=int(raw.get("through_operation_seq") or 0),
            payload=dict(raw.get("payload") or {}),
            replay_tokens=int(raw.get("replay_tokens") or 0),
        )
