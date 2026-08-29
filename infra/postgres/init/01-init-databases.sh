#!/bin/bash
# Creates dedicated databases + least-privilege users for n8n and the AI service.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    -- ---------------------------------------------------------------- n8n ---
    CREATE USER ${N8N_DB_USER} WITH PASSWORD '${N8N_DB_PASSWORD}';
    CREATE DATABASE ${N8N_DB_NAME} OWNER ${N8N_DB_USER};
    GRANT ALL PRIVILEGES ON DATABASE ${N8N_DB_NAME} TO ${N8N_DB_USER};

    -- ------------------------------------------------------- ai-service ---
    CREATE USER ${APP_DB_USER} WITH PASSWORD '${APP_DB_PASSWORD}';
    CREATE DATABASE ${APP_DB_NAME} OWNER ${APP_DB_USER};
    GRANT ALL PRIVILEGES ON DATABASE ${APP_DB_NAME} TO ${APP_DB_USER};
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "${N8N_DB_NAME}" <<-SQL
    GRANT ALL ON SCHEMA public TO ${N8N_DB_USER};
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "${APP_DB_NAME}" <<-SQL
    GRANT ALL ON SCHEMA public TO ${APP_DB_USER};
SQL

echo "[postgres-init] databases '${N8N_DB_NAME}' and '${APP_DB_NAME}' ready"
