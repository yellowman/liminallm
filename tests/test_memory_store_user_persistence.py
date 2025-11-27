from pathlib import Path

from liminallm.storage.memory import MemoryStore
from liminallm.storage.models import AdapterRouterState


def test_memory_store_persists_user_role_and_tenant(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    user = store.create_user(
        "persist@example.com",
        tenant_id="tenant-custom",
        role="admin",
        handle="persist",
    )
    session = store.create_session(user.id, tenant_id="tenant-custom")

    reloaded = MemoryStore(fs_root=str(tmp_path))

    reloaded_user = reloaded.get_user(user.id)
    assert reloaded_user
    assert reloaded_user.role == "admin"
    assert reloaded_user.tenant_id == "tenant-custom"

    reloaded_session = reloaded.get_session(session.id)
    assert reloaded_session
    assert reloaded_session.tenant_id == "tenant-custom"


def test_delete_user_cleans_router_state_and_files(tmp_path):
    store = MemoryStore(fs_root=str(tmp_path))
    user = store.create_user("delete@example.com")

    adapter = store.create_artifact(
        "adapter",
        name="cleanup",
        schema={"kind": "adapter.lora", "backend": "local", "base_model": "stub", "current_version": 0},
        owner_user_id=user.id,
    )
    store.adapter_router_state["state-1"] = AdapterRouterState(artifact_id=adapter.id, base_model="stub")

    user_file = Path(tmp_path) / "users" / user.id / "files" / "upload.txt"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text("uploaded")

    artifact_dir = Path(tmp_path) / "artifacts" / adapter.id
    assert artifact_dir.exists()
    assert user_file.exists()
    assert store.adapter_router_state

    deleted = store.delete_user(user.id)

    assert deleted is True
    assert store.get_user(user.id) is None
    assert not artifact_dir.exists()
    assert not user_file.exists()
    assert not store.adapter_router_state
