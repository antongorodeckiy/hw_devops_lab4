#!/bin/sh
# Wait for Vault to be ready
until VAULT_ADDR=http://vault:8200 vault status 2>/dev/null; do
  echo "Waiting for Vault..."
  sleep 1
done

# Write PostgreSQL credentials to Vault (KV v2 is default in dev mode)
VAULT_ADDR=http://vault:8200 vault kv put secret/db \
  POSTGRES_DB="${POSTGRES_DB}" \
  POSTGRES_USER="${POSTGRES_USER}" \
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  POSTGRES_HOST="${POSTGRES_HOST}" \
  POSTGRES_PORT="${POSTGRES_PORT}"

echo "Vault initialized with DB secrets"
