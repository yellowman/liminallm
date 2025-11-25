from liminallm.storage.memory import MemoryStore


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
