/**
 * The in-app Admin tab: settings console, user management, adapters,
 * storage objects, and config patches. (The standalone /admin page has its
 * own driver in admin.js; this file is the panel inside the chat app.)
 *
 * Definitions only - chat.js wires and invokes these at DOMContentLoaded.
 */

// =============================================================================
// Admin Settings
// =============================================================================

const fetchAdminSettings = async () => {
  if (state.role !== 'admin') return;

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/settings`,
      { method: 'GET', headers: authHeaders() },
      'Failed to load admin settings'
    );

    const data = envelope.data;
    if (data) {
      const defaultPageSize = $('admin-default-page-size');
      const maxPageSize = $('admin-max-page-size');
      const defaultConversationsLimit = $('admin-default-conversations-limit');

      if (defaultPageSize) defaultPageSize.value = data.default_page_size || 100;
      if (maxPageSize) maxPageSize.value = data.max_page_size || 500;
      if (defaultConversationsLimit) defaultConversationsLimit.value = data.default_conversations_limit || 50;
    }
  } catch {
    // Ignore - admin settings are optional
  }
};

const saveAdminSettings = async () => {
  if (state.role !== 'admin') return;

  const statusEl = $('admin-settings-status');
  const defaultPageSize = parseInt($('admin-default-page-size')?.value, 10);
  const maxPageSize = parseInt($('admin-max-page-size')?.value, 10);
  const defaultConversationsLimit = parseInt($('admin-default-conversations-limit')?.value, 10);

  // Validate inputs
  if (isNaN(defaultPageSize) || defaultPageSize < 1 || defaultPageSize > 1000) {
    if (statusEl) statusEl.textContent = 'Invalid default page size (1-1000)';
    return;
  }
  if (isNaN(maxPageSize) || maxPageSize < 1 || maxPageSize > 1000) {
    if (statusEl) statusEl.textContent = 'Invalid max page size (1-1000)';
    return;
  }
  if (isNaN(defaultConversationsLimit) || defaultConversationsLimit < 1 || defaultConversationsLimit > 500) {
    if (statusEl) statusEl.textContent = 'Invalid conversations limit (1-500)';
    return;
  }

  try {
    if (statusEl) statusEl.textContent = 'Saving...';

    await requestEnvelope(
      `${apiBase}/admin/settings`,
      {
        method: 'PATCH',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({
          default_page_size: defaultPageSize,
          max_page_size: maxPageSize,
          default_conversations_limit: defaultConversationsLimit,
        }),
      },
      'Failed to save admin settings'
    );

    if (statusEl) statusEl.textContent = 'Settings saved successfully';
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const renderAdminSettingsSection = () => {
  const section = $('admin-settings-section');
  if (section) {
    section.classList.toggle('hidden', state.role !== 'admin');
  }
  if (state.role === 'admin') {
    fetchAdminSettings();
    fetchAdminUsers();
    fetchAdminAdapters();
    fetchAdminObjects();
    fetchConfigPatches();
  }
};

// =============================================================================
// Admin Users Management
// =============================================================================

const fetchAdminUsers = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('users-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading users...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/users`,
      { headers: authHeaders() },
      'Failed to load users'
    );

    const users = envelope.data?.items || [];
    renderAdminUsers(users);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminUsers = (users) => {
  const tbody = $('users-table-body');
  if (!tbody) return;

  if (!users.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No users found</td></tr>';
    return;
  }

  tbody.innerHTML = users
    .map((user) => {
      const created = user.created_at ? new Date(user.created_at).toLocaleDateString() : '-';
      const isSelf = user.id === state.userId;
      return `
        <tr data-user-id="${escapeHtml(user.id)}">
          <td>${escapeHtml(user.email || '-')}</td>
          <td>
            <select class="user-role-select" ${isSelf ? 'disabled title="Cannot change own role"' : ''}>
              <option value="user" ${user.role === 'user' ? 'selected' : ''}>User</option>
              <option value="admin" ${user.role === 'admin' ? 'selected' : ''}>Admin</option>
            </select>
          </td>
          <td class="monospace">${escapeHtml(user.tenant_id || 'default')}</td>
          <td>${escapeHtml(created)}</td>
          <td>
            <button class="ghost delete-user-btn" ${isSelf ? 'disabled title="Cannot delete self"' : ''}>Delete</button>
          </td>
        </tr>
      `;
    })
    .join('');

  // Add event listeners for role changes
  tbody.querySelectorAll('.user-role-select').forEach((select) => {
    select.addEventListener('change', async (e) => {
      const row = e.target.closest('tr');
      const userId = row?.dataset.userId;
      const newRole = e.target.value;
      if (userId) await changeUserRole(userId, newRole);
    });
  });

  // Add event listeners for delete buttons
  tbody.querySelectorAll('.delete-user-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      const row = e.target.closest('tr');
      const userId = row?.dataset.userId;
      if (userId && confirm('Are you sure you want to delete this user?')) {
        await deleteUser(userId);
      }
    });
  });
};

const createAdminUser = async () => {
  const statusEl = $('create-user-status');
  const email = $('new-user-email')?.value?.trim();
  const password = $('new-user-password')?.value;
  const role = $('new-user-role')?.value || 'user';

  if (!email || !password) {
    if (statusEl) statusEl.textContent = 'Email and password are required';
    return;
  }

  try {
    if (statusEl) statusEl.textContent = 'Creating user...';

    await requestEnvelope(
      `${apiBase}/admin/users`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, role }),
      },
      'Failed to create user'
    );

    if (statusEl) statusEl.textContent = 'User created successfully';
    $('new-user-email').value = '';
    $('new-user-password').value = '';
    $('new-user-role').value = 'user';
    $('add-user-form-section')?.classList.add('hidden');

    fetchAdminUsers();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const changeUserRole = async (userId, newRole) => {
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}/role`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole }),
      },
      'Failed to change user role'
    );

    fetchAdminUsers();
  } catch (err) {
    alert(`Failed to change role: ${err.message}`);
    fetchAdminUsers();
  }
};

const deleteUser = async (userId) => {
  try {
    await requestEnvelope(
      `${apiBase}/admin/users/${userId}`,
      { method: 'DELETE', headers: authHeaders() },
      'Failed to delete user'
    );

    fetchAdminUsers();
  } catch (err) {
    alert(`Failed to delete user: ${err.message}`);
  }
};

// =============================================================================
// Admin Adapters List
// =============================================================================

const fetchAdminAdapters = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('adapters-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading adapters...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/adapters`,
      { headers: authHeaders() },
      'Failed to load adapters'
    );

    const adapters = envelope.data?.items || [];
    renderAdminAdapters(adapters);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminAdapters = (adapters) => {
  const tbody = $('adapters-table-body');
  if (!tbody) return;

  if (!adapters.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No adapters found</td></tr>';
    return;
  }

  tbody.innerHTML = adapters
    .map((adapter) => {
      const created = adapter.created_at ? new Date(adapter.created_at).toLocaleDateString() : '-';
      return `
        <tr>
          <td class="monospace">${escapeHtml(adapter.id?.slice(0, 8) || '-')}...</td>
          <td class="monospace">${escapeHtml(adapter.user_id?.slice(0, 8) || '-')}...</td>
          <td class="monospace">${escapeHtml(adapter.cluster_id?.slice(0, 8) || '-')}...</td>
          <td>${escapeHtml(adapter.base_model_id || '-')}</td>
          <td>${escapeHtml(created)}</td>
        </tr>
      `;
    })
    .join('');
};

// =============================================================================
// Admin Storage Objects
// =============================================================================

const fetchAdminObjects = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('objects-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="empty">Loading objects...</td></tr>';

  try {
    const envelope = await requestEnvelope(
      `${apiBase}/admin/objects`,
      { headers: authHeaders() },
      'Failed to load objects'
    );

    const objects = envelope.data?.items || [];
    renderAdminObjects(objects);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderAdminObjects = (objects) => {
  const tbody = $('objects-table-body');
  if (!tbody) return;

  if (!objects.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No storage objects found</td></tr>';
    return;
  }

  tbody.innerHTML = objects
    .map((obj) => {
      const created = obj.created_at ? new Date(obj.created_at).toLocaleDateString() : '-';
      const sizeDisplay = obj.size_bytes
        ? obj.size_bytes > 1024 * 1024
          ? `${(obj.size_bytes / (1024 * 1024)).toFixed(1)} MB`
          : obj.size_bytes > 1024
          ? `${(obj.size_bytes / 1024).toFixed(1)} KB`
          : `${obj.size_bytes} B`
        : '-';
      return `
        <tr>
          <td class="monospace">${escapeHtml(obj.key || obj.object_key || '-')}</td>
          <td>${escapeHtml(obj.bucket || obj.bucket_name || 'default')}</td>
          <td>${escapeHtml(sizeDisplay)}</td>
          <td>${escapeHtml(obj.content_type || obj.mime_type || '-')}</td>
          <td>${escapeHtml(created)}</td>
        </tr>
      `;
    })
    .join('');
};

// =============================================================================
// Admin Config Patches
// =============================================================================

let selectedPatch = null;

const fetchConfigPatches = async () => {
  if (state.role !== 'admin') return;

  const tbody = $('patches-table-body');
  if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="empty">Loading patches...</td></tr>';

  try {
    const statusFilter = $('patches-status-filter')?.value || '';
    const url = statusFilter
      ? `${apiBase}/config/patches?status=${statusFilter}`
      : `${apiBase}/config/patches`;

    const envelope = await requestEnvelope(url, { headers: authHeaders() }, 'Failed to load patches');

    const patches = envelope.data?.items || [];
    renderConfigPatches(patches);
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="6" class="empty">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
};

const renderConfigPatches = (patches) => {
  const tbody = $('patches-table-body');
  if (!tbody) return;

  if (!patches.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No config patches found</td></tr>';
    $('patch-details-section')?.classList.add('hidden');
    return;
  }

  tbody.innerHTML = patches
    .map((patch) => {
      const created = patch.created_at ? new Date(patch.created_at).toLocaleDateString() : '-';
      const statusClass = patch.status === 'pending' ? 'pending' : patch.status === 'approved' ? 'approved' : patch.status === 'applied' ? 'applied' : 'rejected';
      return `
        <tr class="clickable" data-patch-id="${patch.id}">
          <td>${escapeHtml(String(patch.id))}</td>
          <td class="monospace">${escapeHtml(patch.artifact_id?.slice(0, 8) || '-')}...</td>
          <td>${escapeHtml(patch.proposer || '-')}</td>
          <td><span class="patch-status ${statusClass}">${escapeHtml(patch.status)}</span></td>
          <td>${escapeHtml(created)}</td>
          <td>
            <button class="ghost view-patch-btn" data-patch-id="${patch.id}">View</button>
          </td>
        </tr>
      `;
    })
    .join('');

  // Add click handlers
  tbody.querySelectorAll('.view-patch-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const patchId = btn.dataset.patchId;
      const patch = patches.find((p) => String(p.id) === patchId);
      if (patch) selectPatch(patch);
    });
  });

  tbody.querySelectorAll('tr.clickable').forEach((row) => {
    row.addEventListener('click', () => {
      const patchId = row.dataset.patchId;
      const patch = patches.find((p) => String(p.id) === patchId);
      if (patch) selectPatch(patch);
    });
  });
};

const selectPatch = (patch) => {
  selectedPatch = patch;
  const detailsSection = $('patch-details-section');
  const detailsContent = $('patch-details-content');
  const approveBtn = $('approve-patch-btn');
  const rejectBtn = $('reject-patch-btn');
  const applyBtn = $('apply-patch-btn');

  if (!detailsSection || !detailsContent) return;

  detailsSection.classList.remove('hidden');

  // Render patch details
  const patchJson = JSON.stringify(patch.patch || {}, null, 2);
  detailsContent.innerHTML = `
    <div class="detail-row"><span class="detail-label">Patch ID</span><span>${escapeHtml(String(patch.id))}</span></div>
    <div class="detail-row"><span class="detail-label">Artifact</span><span class="monospace">${escapeHtml(patch.artifact_id || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">Proposer</span><span>${escapeHtml(patch.proposer || '-')}</span></div>
    <div class="detail-row"><span class="detail-label">Status</span><span class="patch-status ${patch.status}">${escapeHtml(patch.status)}</span></div>
    <div class="detail-row"><span class="detail-label">Justification</span><span>${escapeHtml(patch.justification || 'None provided')}</span></div>
    <div class="detail-row"><span class="detail-label">Created</span><span>${patch.created_at ? new Date(patch.created_at).toLocaleString() : '-'}</span></div>
    ${patch.decided_at ? `<div class="detail-row"><span class="detail-label">Decided</span><span>${new Date(patch.decided_at).toLocaleString()}</span></div>` : ''}
    ${patch.applied_at ? `<div class="detail-row"><span class="detail-label">Applied</span><span>${new Date(patch.applied_at).toLocaleString()}</span></div>` : ''}
    <div class="patch-code">
      <label>Patch Content</label>
      <pre class="code-block">${escapeHtml(patchJson)}</pre>
    </div>
  `;

  // Update button states based on status
  if (approveBtn) approveBtn.disabled = patch.status !== 'pending';
  if (rejectBtn) rejectBtn.disabled = patch.status !== 'pending';
  if (applyBtn) applyBtn.disabled = patch.status !== 'approved';

  // Clear any previous status
  const statusEl = $('patch-action-status');
  if (statusEl) statusEl.textContent = '';
};

const decidePatch = async (decision) => {
  if (!selectedPatch) return;

  const statusEl = $('patch-action-status');

  try {
    if (statusEl) statusEl.textContent = `${decision === 'approve' ? 'Approving' : 'Rejecting'} patch...`;

    await requestEnvelope(
      `${apiBase}/config/patches/${selectedPatch.id}/decide`,
      {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      },
      `Failed to ${decision} patch`
    );

    if (statusEl) statusEl.textContent = `Patch ${decision}d successfully`;
    fetchConfigPatches();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

const applyPatch = async () => {
  if (!selectedPatch || selectedPatch.status !== 'approved') return;

  const statusEl = $('patch-action-status');

  try {
    if (statusEl) statusEl.textContent = 'Applying patch...';

    await requestEnvelope(
      `${apiBase}/config/patches/${selectedPatch.id}/apply`,
      {
        method: 'POST',
        headers: authHeaders(),
      },
      'Failed to apply patch'
    );

    if (statusEl) statusEl.textContent = 'Patch applied successfully';
    fetchConfigPatches();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Error: ${err.message}`;
  }
};

