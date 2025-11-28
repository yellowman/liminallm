import json

from liminallm.service.clustering import SemanticClusterer
import asyncio

from liminallm.storage.models import PreferenceEvent, SemanticCluster


class RecordingLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts = []

    def generate(self, prompt, adapters, context_snippets, history=None):
        self.prompts.append(prompt)
        return {"content": self.content}


class StubStore:
    def __init__(self) -> None:
        self.updates = []

    def update_semantic_cluster(
        self, cluster_id, *, label=None, description=None, meta=None
    ):
        self.updates.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "description": description,
                "meta": meta,
            }
        )


CLUSTER = SemanticCluster(
    id="cluster-1",
    user_id="user-1",
    centroid=[0.1, 0.2],
    size=2,
    sample_message_ids=[],
)


EVENTS = [
    PreferenceEvent(
        id="evt-1",
        user_id="user-1",
        conversation_id="conv-1",
        message_id="msg-1",
        feedback="positive",
        context_embedding=[0.1, 0.1],
        cluster_id="cluster-1",
        context_text="Write a Rust parser",
    ),
    PreferenceEvent(
        id="evt-2",
        user_id="user-1",
        conversation_id="conv-1",
        message_id="msg-2",
        feedback="positive",
        context_embedding=[0.1, 0.2],
        cluster_id="cluster-1",
        corrected_text="Make it handle errors gracefully",
    ),
]


def test_label_clusters_uses_llm_json_response():
    store = StubStore()
    llm = RecordingLLM(
        json.dumps({"label": "Rust helpers", "description": "Helps write Rust code."})
    )
    clusterer = SemanticClusterer(store, llm=llm)

    asyncio.run(clusterer.label_clusters([CLUSTER], EVENTS))

    assert store.updates[0]["label"] == "Rust helpers"
    assert store.updates[0]["description"] == "Helps write Rust code."
    assert any("Return JSON" in prompt for prompt in llm.prompts)


def test_label_clusters_falls_back_without_llm_response():
    store = StubStore()
    llm = RecordingLLM("")
    clusterer = SemanticClusterer(store, llm=llm)

    asyncio.run(clusterer.label_clusters([CLUSTER], EVENTS))

    assert store.updates[0]["label"].startswith("Write a Rust")
    assert "labeled_at" in (store.updates[0]["meta"] or {})
