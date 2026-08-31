/**
 * Auth flows for the chat page: login/signup form switching, session and
 * token management, OAuth, password reset, and the email-token URL handlers.
 *
 * Definitions only - chat.js wires these to the DOM in initEventListeners()
 * and init(), which run at DOMContentLoaded, after every deferred script.
 */

// =============================================================================
// Auth form switching
// =============================================================================

const showAuthForm = (formName) => {
  const loginContainer = $('login-form-container');
  const signupContainer = $('signup-form-container');
  const resetContainer = $('reset-form-container');

  if (loginContainer) loginContainer.classList.toggle('hidden', formName !== 'login');
  if (signupContainer) signupContainer.classList.toggle('hidden', formName !== 'signup');
  if (resetContainer) resetContainer.classList.toggle('hidden', formName !== 'reset');
};

// =============================================================================
// Auth management
// =============================================================================

// An admin gets one extra control in the composer's own button row rather
// than a banner: the banner said what the row can show, and cost the chat
// column a line of vertical space on every turn to say it.
const renderAdminNotice = () => {
  const isAdmin = state.role === 'admin';
  if (approvePatches) approvePatches.classList.toggle('hidden', !isAdmin);
  // A class, so the stylesheet keeps deciding how a rail button lays out.
  // Setting `display: inline-flex` here overrode `.rail-btn`'s own `grid`,
  // and `place-items: center` centres nothing in a flex row - the shield sat
  // 8px left of every other icon, the one item in the rail out of line.
  if (adminLink) adminLink.classList.toggle('hidden', !isAdmin);
  // Show/hide admin settings section based on role
  renderAdminSettingsSection();
};

// Neither the refresh token nor the session id is kept: both are HttpOnly
// cookies the page cannot read (SPEC §17.10), and a copy here would be a
// durable credential in reach of any script on the page.
const persistAuth = (payload) => {
  state.accessToken = payload.access_token;
  state.role = payload.role;
  state.tenantId = payload.tenant_id;
  state.userId = payload.user_id;
  updateAuthUI();
  renderAdminNotice();
  refreshUploadLimits();
};

const handleLogin = async (event) => {
  event.preventDefault();
  const body = {
    email: $('email')?.value,
    password: $('password')?.value,
    mfa_code: $('mfa')?.value || undefined,
  };

  try {
    toggleButtonBusy(authSubmit, true, 'Signing in...');
    const envelope = await requestEnvelope(
      `${apiBase}/auth/login`,
      { method: 'POST', headers: jsonHeaders(), body: JSON.stringify(body) },
      'Login failed'
    );

    if (envelope.data?.mfa_required && !envelope.data?.access_token) {
      showStatus('MFA required. Enter the code from your authenticator.', true);
      return;
    }

    persistAuth(envelope.data);
    showStatus('Signed in');

    // Load initial data
    await Promise.all([
      fetchConversations(),
      fetchContexts(),
      fetchArtifacts(),
      fetchHealth(),
      fetchMfaStatus(),
      fetchEmailVerificationStatus(),
      fetchUserSettings(),
    ]);
  } catch (err) {
    showStatus(err.message, true);
  } finally {
    toggleButtonBusy(authSubmit, false);
  }
};

const logout = async () => {
  try {
    await requestEnvelope(
      `${apiBase}/auth/logout`,
      { method: 'POST', headers: headers(), keepalive: true },
      'Logout failed'
    );
  } catch {
    // Continue with local cleanup
  }

  state.resetAuth();
  setConversation(null);
  if (messagesEl) messagesEl.innerHTML = '';
  updateAuthUI();
  renderAdminNotice();
  updateEmptyState();
};

// =============================================================================
// OAuth
// =============================================================================

// Allowed OAuth providers (prevents path traversal via malicious provider values)
const ALLOWED_OAUTH_PROVIDERS = new Set(['google', 'github', 'microsoft']);

const validateOAuthProvider = (provider) => {
  if (!provider || typeof provider !== 'string') return false;
  return ALLOWED_OAUTH_PROVIDERS.has(provider.toLowerCase());
};

const startOAuth = async (provider) => {
  const oauthStatus = $('oauth-status');
  const btn = $(`oauth-${provider}`);

  try {
    if (btn) btn.disabled = true;
    if (oauthStatus) oauthStatus.textContent = `Connecting to ${provider}...`;

    const redirectUri = window.location.origin + window.location.pathname;

    const envelope = await requestEnvelope(
      `${apiBase}/auth/oauth/${provider}/start`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ redirect_uri: redirectUri }),
      },
      `Failed to start ${provider} login`
    );

    const authUrl = envelope.data?.authorization_url;
    if (authUrl) {
      // Store the state for callback verification
      sessionStorage.setItem('oauth_state', envelope.data?.state || '');
      sessionStorage.setItem('oauth_provider', provider);
      // Redirect to OAuth provider
      window.location.href = authUrl;
    } else {
      if (oauthStatus) oauthStatus.textContent = 'No authorization URL returned';
    }
  } catch (err) {
    if (oauthStatus) oauthStatus.textContent = err.message;
  } finally {
    if (btn) btn.disabled = false;
  }
};

const handleOAuthCallback = async () => {
  // Check if this is an OAuth callback
  const urlParams = new URLSearchParams(window.location.search);
  const code = urlParams.get('code');
  const state = urlParams.get('state');
  const provider = urlParams.get('provider') || sessionStorage.getItem('oauth_provider');

  if (!code || !state || !provider) {
    return false;
  }

  const oauthStatus = $('oauth-status');

  // Validate provider to prevent path traversal attacks (Issue 41.4)
  if (!validateOAuthProvider(provider)) {
    if (oauthStatus) oauthStatus.textContent = 'Invalid OAuth provider.';
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');
    // Clear URL params
    window.history.replaceState({}, document.title, window.location.pathname);
    return true;
  }

  const storedState = sessionStorage.getItem('oauth_state');

  // Verify state matches
  if (storedState && state !== storedState) {
    if (oauthStatus) oauthStatus.textContent = 'OAuth state mismatch. Please try again.';
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');
    return true;
  }

  try {
    if (oauthStatus) oauthStatus.textContent = 'Completing sign in...';


    const envelope = await requestEnvelope(
      `${apiBase}/auth/oauth/${provider}/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`,
      {
        method: 'GET',
        headers: jsonHeaders(),
      },
      `OAuth callback failed`
    );

    // Clear OAuth session data
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');

    // Clear URL params
    window.history.replaceState({}, document.title, window.location.pathname);

    if (envelope.data?.access_token) {
      persistAuth(envelope.data);
      showStatus('Signed in with ' + provider);

      // Load initial data
      await Promise.all([
        fetchConversations(),
        fetchContexts(),
        fetchArtifacts(),
        fetchHealth(),
        fetchMfaStatus(),
        fetchEmailVerificationStatus(),
        fetchUserSettings(),
      ]);
    } else {
      if (oauthStatus) oauthStatus.textContent = 'OAuth completed but no token received';
    }
  } catch (err) {
    if (oauthStatus) oauthStatus.textContent = err.message;
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('oauth_provider');
  }

  return true;
};

// =============================================================================
// Signup
// =============================================================================

const handleSignup = async (event) => {
  event.preventDefault();
  const signupStatus = $('signup-status');
  const signupSubmit = $('signup-submit');

  const email = $('signup-email')?.value?.trim();
  const password = $('signup-password')?.value;
  const confirm = $('signup-confirm')?.value;

  if (!email || !password) {
    if (signupStatus) signupStatus.textContent = 'Email and password are required';
    return;
  }

  if (password !== confirm) {
    if (signupStatus) signupStatus.textContent = 'Passwords do not match';
    return;
  }

  if (password.length < 8) {
    if (signupStatus) signupStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  try {
    toggleButtonBusy(signupSubmit, true, 'Creating...');
    if (signupStatus) signupStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/signup`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ email, password }),
      },
      'Signup failed'
    );

    if (signupStatus) signupStatus.textContent = 'Account created! You can now sign in.';

    // Clear form and switch to login
    if ($('signup-email')) $('signup-email').value = '';
    if ($('signup-password')) $('signup-password').value = '';
    if ($('signup-confirm')) $('signup-confirm').value = '';

    // Pre-fill login email
    if ($('email')) $('email').value = email;

    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (signupStatus) signupStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(signupSubmit, false);
  }
};

// =============================================================================
// Password Reset
// =============================================================================

let resetEmailForConfirm = '';

const handleResetRequest = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const resetSubmit = $('reset-request-submit');
  const resetCodeSection = $('reset-code-section');

  const email = $('reset-email')?.value?.trim();

  if (!email) {
    if (resetStatus) resetStatus.textContent = 'Email is required';
    return;
  }

  try {
    toggleButtonBusy(resetSubmit, true, 'Sending...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/request`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ email }),
      },
      'Reset request failed'
    );

    resetEmailForConfirm = email;
    if (resetStatus) resetStatus.textContent = 'Reset code sent! Check your email.';
    if (resetCodeSection) resetCodeSection.classList.remove('hidden');
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(resetSubmit, false);
  }
};

const handleResetConfirm = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const confirmSubmit = $('reset-confirm-submit');

  const code = $('reset-code')?.value?.trim();
  const newPassword = $('reset-new-password')?.value;

  if (!code || !newPassword) {
    if (resetStatus) resetStatus.textContent = 'Code and new password are required';
    return;
  }

  if (newPassword.length < 8) {
    if (resetStatus) resetStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  try {
    toggleButtonBusy(confirmSubmit, true, 'Resetting...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/confirm`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({
          token: code,
          new_password: newPassword,
        }),
      },
      'Password reset failed'
    );

    if (resetStatus) resetStatus.textContent = 'Password reset successful! You can now sign in.';

    // Clear form
    if ($('reset-code')) $('reset-code').value = '';
    if ($('reset-new-password')) $('reset-new-password').value = '';
    if ($('reset-code-section')) $('reset-code-section').classList.add('hidden');

    // Pre-fill login email
    if ($('email')) $('email').value = resetEmailForConfirm;

    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(confirmSubmit, false);
  }
};

// =============================================================================
// Email Token URL Handlers
// =============================================================================

let pendingResetToken = '';

const handleResetTokenCallback = async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const resetToken = urlParams.get('reset_token');

  if (!resetToken) {
    return false;
  }

  // Clear URL params
  window.history.replaceState({}, document.title, window.location.pathname);

  // Store the token for the form
  pendingResetToken = resetToken;

  // Show the reset form with just the new password field visible
  showAuthForm('reset');

  const resetStatus = $('reset-status');
  const resetCodeSection = $('reset-code-section');
  const resetRequestForm = $('reset-request-form');

  // Hide the email request section and show the password input
  if (resetRequestForm) resetRequestForm.classList.add('hidden');
  if (resetCodeSection) {
    resetCodeSection.classList.remove('hidden');
    // Hide the code input since we have the token from URL
    const codeField = resetCodeSection.querySelector('.field:first-child');
    if (codeField) codeField.classList.add('hidden');
  }

  // Update the description
  const subtext = resetCodeSection?.previousElementSibling;
  if (subtext && subtext.classList.contains('subtext')) {
    subtext.textContent = 'Enter your new password.';
  }

  if (resetStatus) resetStatus.textContent = 'Password reset link verified. Enter your new password.';

  return true;
};

const handleResetWithToken = async (event) => {
  event.preventDefault();
  const resetStatus = $('reset-status');
  const confirmSubmit = $('reset-confirm-submit');

  const newPassword = $('reset-new-password')?.value;

  if (!newPassword) {
    if (resetStatus) resetStatus.textContent = 'New password is required';
    return;
  }

  if (newPassword.length < 8) {
    if (resetStatus) resetStatus.textContent = 'Password must be at least 8 characters';
    return;
  }

  if (!pendingResetToken) {
    if (resetStatus) resetStatus.textContent = 'Reset token expired. Please request a new reset link.';
    return;
  }

  try {
    toggleButtonBusy(confirmSubmit, true, 'Resetting...');
    if (resetStatus) resetStatus.textContent = '';

    await requestEnvelope(
      `${apiBase}/auth/reset/confirm`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({
          token: pendingResetToken,
          new_password: newPassword,
        }),
      },
      'Password reset failed'
    );

    pendingResetToken = '';
    if (resetStatus) resetStatus.textContent = 'Password reset successful! You can now sign in.';

    // Clear form and show login after delay
    if ($('reset-new-password')) $('reset-new-password').value = '';
    setTimeout(() => showAuthForm('login'), 1500);
  } catch (err) {
    if (resetStatus) resetStatus.textContent = err.message;
  } finally {
    toggleButtonBusy(confirmSubmit, false);
  }
};

const handleVerifyTokenCallback = async () => {
  const urlParams = new URLSearchParams(window.location.search);
  const verifyToken = urlParams.get('verify_token');

  if (!verifyToken) {
    return false;
  }

  // Clear URL params
  window.history.replaceState({}, document.title, window.location.pathname);

  // Show a verification status message
  showStatus('Verifying email...');

  try {
    await requestEnvelope(
      `${apiBase}/auth/verify_email`,
      {
        method: 'POST',
        headers: jsonHeaders(),
        body: JSON.stringify({ token: verifyToken }),
      },
      'Email verification failed'
    );

    showStatus('Email verified successfully! You can now sign in.');

    // Show login form
    showAuthForm('login');
  } catch (err) {
    showStatus(`Email verification failed: ${err.message}`);
  }

  return true;
};

