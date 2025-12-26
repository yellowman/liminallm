#!/usr/bin/env bash
# QA Smoke Test Script for LiminalLM
#
# This script performs end-to-end smoke tests against a running LiminalLM instance.
# It tests authentication, basic API functionality, and access control.
#
# Usage:
#   ./scripts/smoke_test.sh                    # Test against localhost:8000
#   ./scripts/smoke_test.sh http://app:8000   # Test against custom URL
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed

set -euo pipefail

# Configuration
BASE_URL="${1:-http://localhost:8000}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@test.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-TestAdmin123!}"
TEST_USER_EMAIL="smoketest_$(date +%s)@test.local"
TEST_USER_PASSWORD="SmokeTest123!"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

run_test() {
    local name="$1"
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n--- Test $TESTS_RUN: $name ---"
}

# Check if jq is available
check_dependencies() {
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is required but not installed"
        exit 1
    fi
    if ! command -v jq &> /dev/null; then
        echo "Warning: jq not found, JSON parsing will be limited"
        JQ_AVAILABLE=false
    else
        JQ_AVAILABLE=true
    fi
}

# Wait for service to be ready
wait_for_service() {
    log_info "Waiting for service at $BASE_URL..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$BASE_URL/healthz" > /dev/null 2>&1; then
            log_info "Service is ready"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    echo ""
    log_fail "Service did not become ready within timeout"
    exit 1
}

# Extract JSON field (works with or without jq)
extract_json() {
    local json="$1"
    local field="$2"
    if $JQ_AVAILABLE; then
        echo "$json" | jq -r "$field" 2>/dev/null || echo ""
    else
        # Basic extraction for common patterns
        echo "$json" | grep -oP "\"$field\":\s*\"[^\"]*\"" | sed 's/.*"\([^"]*\)"$/\1/' || echo ""
    fi
}

#######################
# Test Cases
#######################

test_healthz() {
    run_test "Health check endpoint"

    local response
    local http_code

    response=$(curl -sf "$BASE_URL/healthz" 2>&1) || {
        log_fail "Health check request failed"
        return 1
    }

    if echo "$response" | grep -q '"status"'; then
        log_pass "Health check returned valid response"
        return 0
    else
        log_fail "Health check response invalid: $response"
        return 1
    fi
}

test_signup_user() {
    run_test "User signup"

    local response
    local http_code

    response=$(curl -sf -X POST "$BASE_URL/v1/auth/signup" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$TEST_USER_EMAIL\", \"password\": \"$TEST_USER_PASSWORD\"}" 2>&1) || {
        log_fail "Signup request failed"
        return 1
    }

    USER_ACCESS_TOKEN=$(extract_json "$response" ".data.access_token")
    USER_ID=$(extract_json "$response" ".data.user_id")

    if [ -n "$USER_ACCESS_TOKEN" ] && [ "$USER_ACCESS_TOKEN" != "null" ]; then
        log_pass "User signup successful (user_id: $USER_ID)"
        return 0
    else
        log_fail "Signup failed: $response"
        return 1
    fi
}

test_login_user() {
    run_test "User login"

    local response

    response=$(curl -sf -X POST "$BASE_URL/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$TEST_USER_EMAIL\", \"password\": \"$TEST_USER_PASSWORD\"}" 2>&1) || {
        log_fail "Login request failed"
        return 1
    }

    local token
    token=$(extract_json "$response" ".data.access_token")

    if [ -n "$token" ] && [ "$token" != "null" ]; then
        USER_ACCESS_TOKEN="$token"
        log_pass "User login successful"
        return 0
    else
        log_fail "Login failed: $response"
        return 1
    fi
}

test_login_admin() {
    run_test "Admin login"

    local response

    response=$(curl -sf -X POST "$BASE_URL/v1/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$ADMIN_EMAIL\", \"password\": \"$ADMIN_PASSWORD\"}" 2>&1) || {
        log_fail "Admin login request failed"
        return 1
    }

    ADMIN_ACCESS_TOKEN=$(extract_json "$response" ".data.access_token")

    if [ -n "$ADMIN_ACCESS_TOKEN" ] && [ "$ADMIN_ACCESS_TOKEN" != "null" ]; then
        log_pass "Admin login successful"
        return 0
    else
        log_fail "Admin login failed: $response"
        return 1
    fi
}

test_create_conversation() {
    run_test "Create conversation"

    local response

    response=$(curl -sf -X POST "$BASE_URL/v1/conversations" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"title": "Smoke Test Conversation"}' 2>&1) || {
        log_fail "Create conversation request failed"
        return 1
    }

    CONVERSATION_ID=$(extract_json "$response" ".data.id")

    if [ -n "$CONVERSATION_ID" ] && [ "$CONVERSATION_ID" != "null" ]; then
        log_pass "Conversation created (id: $CONVERSATION_ID)"
        return 0
    else
        log_fail "Create conversation failed: $response"
        return 1
    fi
}

test_chat_message() {
    run_test "Send chat message"

    if [ -z "${CONVERSATION_ID:-}" ]; then
        log_fail "No conversation ID available (previous test failed)"
        return 1
    fi

    local response
    local http_code
    local tmpfile
    tmpfile=$(mktemp)

    # ChatRequest schema requires: conversation_id, message: {content, mode}, stream
    # With MODEL_BACKEND=stub, this should always return 200 with a canned response
    http_code=$(curl -s -w "%{http_code}" -o "$tmpfile" -X POST "$BASE_URL/v1/chat" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"conversation_id\": \"$CONVERSATION_ID\", \"message\": {\"content\": \"Hello, this is a smoke test.\", \"mode\": \"text\"}, \"stream\": false}" 2>&1)

    response=$(cat "$tmpfile")
    rm -f "$tmpfile"

    # Require 200 - stub backend should always succeed
    # Do NOT accept 422 (malformed request) or 503 (no backend)
    if [ "$http_code" = "200" ]; then
        # Verify response contains content from stub backend
        if echo "$response" | grep -q "stub response"; then
            log_pass "Chat endpoint returned stub response (status: 200)"
        else
            log_pass "Chat endpoint successful (status: 200)"
        fi
        return 0
    else
        log_fail "Chat request failed with status $http_code: $response"
        return 1
    fi
}

test_file_upload() {
    run_test "File upload"

    # Create a temporary test file
    local test_file
    test_file=$(mktemp)
    echo "Smoke test file content - $(date)" > "$test_file"

    local response
    local http_code
    local tmpfile
    tmpfile=$(mktemp)

    # Correct endpoint is /v1/files/upload (not /v1/files which is GET-only)
    http_code=$(curl -s -w "%{http_code}" -o "$tmpfile" -X POST "$BASE_URL/v1/files/upload" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" \
        -F "file=@$test_file;filename=smoke_test.txt;type=text/plain" 2>&1)

    response=$(cat "$tmpfile")
    rm -f "$test_file" "$tmpfile"

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        log_pass "File upload successful"
        return 0
    else
        log_fail "File upload failed with status $http_code: $response"
        return 1
    fi
}

test_file_limits() {
    run_test "File limits endpoint"

    local response
    local http_code
    local tmpfile
    tmpfile=$(mktemp)

    http_code=$(curl -s -w "%{http_code}" -o "$tmpfile" -X GET "$BASE_URL/v1/files/limits" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" 2>&1)

    response=$(cat "$tmpfile")
    rm -f "$tmpfile"

    if [ "$http_code" = "200" ]; then
        # Verify response contains expected fields
        if echo "$response" | grep -q "allowed_extensions"; then
            log_pass "File limits endpoint returns allowed_extensions"
            return 0
        else
            log_fail "File limits response missing allowed_extensions: $response"
            return 1
        fi
    else
        log_fail "File limits failed with status $http_code: $response"
        return 1
    fi
}

test_admin_settings_protected() {
    run_test "Admin settings protected from regular user"

    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL/v1/admin/settings" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" 2>&1)

    if [ "$http_code" = "403" ]; then
        log_pass "Admin settings correctly returns 403 for regular user"
        return 0
    else
        log_fail "Admin settings returned $http_code instead of 403 for regular user"
        return 1
    fi
}

test_admin_settings_accessible() {
    run_test "Admin settings accessible to admin"

    if [ -z "${ADMIN_ACCESS_TOKEN:-}" ]; then
        log_fail "No admin token available (admin login failed)"
        return 1
    fi

    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL/v1/admin/settings" \
        -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" 2>&1)

    if [ "$http_code" = "200" ]; then
        log_pass "Admin settings accessible to admin (status: 200)"
        return 0
    else
        log_fail "Admin settings returned $http_code instead of 200 for admin"
        return 1
    fi
}

test_list_artifacts() {
    run_test "List artifacts"

    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL/v1/artifacts" \
        -H "Authorization: Bearer $USER_ACCESS_TOKEN" 2>&1)

    if [ "$http_code" = "200" ]; then
        log_pass "List artifacts successful"
        return 0
    else
        log_fail "List artifacts failed with status $http_code"
        return 1
    fi
}

test_unauthenticated_protected() {
    run_test "Protected endpoints require authentication"

    local http_code

    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL/v1/conversations" 2>&1)

    if [ "$http_code" = "401" ]; then
        log_pass "Protected endpoint correctly returns 401 for unauthenticated request"
        return 0
    else
        log_fail "Protected endpoint returned $http_code instead of 401"
        return 1
    fi
}

#######################
# Main
#######################

main() {
    echo "========================================"
    echo "LiminalLM QA Smoke Test"
    echo "========================================"
    echo "Base URL: $BASE_URL"
    echo "Admin Email: $ADMIN_EMAIL"
    echo "Test User: $TEST_USER_EMAIL"
    echo "========================================"

    check_dependencies
    wait_for_service

    # Run tests (use || true to continue on test failures with set -e)
    test_healthz || true
    test_unauthenticated_protected || true
    test_signup_user || true
    test_login_user || true
    test_login_admin || true
    test_create_conversation || true
    test_chat_message || true
    test_file_upload || true
    test_file_limits || true
    test_list_artifacts || true
    test_admin_settings_protected || true
    test_admin_settings_accessible || true

    # Summary
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Tests Run: $TESTS_RUN"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo "========================================"

    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}SMOKE TEST FAILED${NC}"
        exit 1
    else
        echo -e "${GREEN}SMOKE TEST PASSED${NC}"
        exit 0
    fi
}

main "$@"
