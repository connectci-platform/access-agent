#!/usr/bin/env bash
#
# vault-init.sh — Bootstrap Vault for ACCESS agent JWT authentication.
#
# Usage:
#   ./scripts/vault-init.sh
#
# Prerequisites:
#   - Vault is running (e.g. via docker-compose)
#   - VAULT_ADDR and VAULT_TOKEN are set (or defaults are used)
#
# What this script does:
#   1. Waits for Vault to be ready
#   2. Enables the KV v2 secrets engine (if not already enabled)
#   3. Stores a JWT signing secret at secret/access/jwt
#   4. Creates a read-only policy for the agent
#   5. Creates an AppRole for the agent to authenticate
#
set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
# Default matches VAULT_DEV_ROOT_TOKEN_ID in docker-compose.yml.
# When using the Vault dev server, VAULT_TOKEN can be left unset.
VAULT_TOKEN="${VAULT_TOKEN:-dev-token}"
SECRET_PATH="access/jwt"
POLICY_NAME="access-agent-jwt"
APPROLE_NAME="access-agent"

export VAULT_ADDR VAULT_TOKEN

echo "==> Waiting for Vault at ${VAULT_ADDR}..."
for i in $(seq 1 30); do
  if vault status >/dev/null 2>&1; then
    echo "    Vault is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Vault did not become ready in time." >&2
    exit 1
  fi
  sleep 1
done

# Enable KV v2 secrets engine (idempotent — ignores "already enabled" error)
echo "==> Enabling KV v2 secrets engine..."
vault secrets enable -version=2 secret 2>/dev/null || true

# Generate a random signing key if one isn't provided
JWT_SIGNING_KEY="${JWT_SIGNING_KEY:-$(openssl rand -base64 32)}"

echo "==> Storing JWT signing secret at secret/${SECRET_PATH}..."
vault kv put "secret/${SECRET_PATH}" signing_key="${JWT_SIGNING_KEY}"

# Create a policy that allows reading the JWT secret
echo "==> Creating policy '${POLICY_NAME}'..."
vault policy write "${POLICY_NAME}" - <<EOF
# Allow reading the JWT signing secret
path "secret/data/${SECRET_PATH}" {
  capabilities = ["read"]
}
EOF

# Enable AppRole auth (idempotent)
echo "==> Enabling AppRole auth method..."
vault auth enable approle 2>/dev/null || true

# Create an AppRole for the agent
echo "==> Creating AppRole '${APPROLE_NAME}'..."
vault write "auth/approle/role/${APPROLE_NAME}" \
  token_policies="${POLICY_NAME}" \
  token_ttl=1h \
  token_max_ttl=4h

# Fetch the role ID and secret ID for the agent
ROLE_ID=$(vault read -field=role_id "auth/approle/role/${APPROLE_NAME}/role-id")
SECRET_ID=$(vault write -f -field=secret_id "auth/approle/role/${APPROLE_NAME}/secret-id")

echo ""
echo "==> Vault initialization complete."
echo ""
echo "    AppRole credentials for the agent:"
echo "    VAULT_ROLE_ID=${ROLE_ID}"
echo "    VAULT_SECRET_ID=${SECRET_ID}"
echo ""
echo "    For development, you can also use the root token:"
echo "    VAULT_TOKEN=${VAULT_TOKEN}"
echo ""
echo "    JWT signing key stored at: secret/${SECRET_PATH}"
