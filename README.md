# ResearchLab

A portfolio AI engineering workspace with independent RAG and Agentic execution modes. **Milestone 1: foundation only.** All AI capabilities and Compare results are clearly marked as unimplemented. No paid services are called.

## Requirements

Python 3.12+, Node.js 22.12+ with npm, and Docker Compose v2. Python 3.14 and Node 24 are available in the implementation environment. Docker is not installed there, so live database validation remains required.

## Local development

From the repository root (PowerShell):

```powershell
Copy-Item .env.example .env
```

Set your own nonempty `POSTGRES_PASSWORD` in `.env`. Set `DATABASE_URL` to `postgresql+psycopg://<user>:<URL-encoded-password>@localhost:5432/<database>`, matching the PostgreSQL variables. URL-encode reserved characters in the URL password, keeping the raw password in `POSTGRES_PASSWORD`. Match the URL port to `POSTGRES_PORT`. No credential defaults are provided; never commit `.env`.

```powershell
docker compose up -d --wait
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./backend[dev]"
.\.venv\Scripts\alembic.exe -c backend/alembic.ini upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --no-access-log
```

In another terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open http://localhost:5173. API docs: http://127.0.0.1:8000/docs. Vite proxies `/api` to port 8000. `npm.cmd` avoids restrictive PowerShell script execution policies. On macOS/Linux use `npm`, `.venv/bin/python` and `.venv/bin/alembic`.

The API can start with no database URL for isolated liveness/configuration tests. Database operations and migrations require a URL. Process environment overrides the root `.env`. `APP_ENV`, `LOG_LEVEL`, JSON-list `CORS_ORIGINS`, and optional secret `DATABASE_URL` are validated through Pydantic Settings. No LLM keys are needed.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

After starting PostgreSQL and migrating, verify the extension from the root:

```powershell
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname, extversion FROM pg_extension;"'
.\.venv\Scripts\alembic.exe -c backend/alembic.ini current
```

Expect a `vector` row and revision `0001_enable_vector`. For offline migration SQL, set `DATABASE_URL` and run `alembic -c backend/alembic.ini upgrade head --sql` using the virtual environment executable. Offline generation does not verify a live database.

## Architecture

```text
backend/
  app/
    api/              HTTP routes and request-scoped session dependency
    application/      LLM, embedding and run repository protocols
    domain/           Framework-independent mode and run types
    infrastructure/   Settings, JSON logging, SQLAlchemy
    main.py           Application factory and dependency composition
  migrations/         Alembic environment and versioned migration
  tests/              Health, CORS, logging and configuration
frontend/
  src/                React shell, mode navigation and real API health status
compose.yaml          PostgreSQL 17 + pgvector with persistent volume
```

This pragmatic modular monolith keeps operations simple and defines seams for independent future RAG and Agentic tests. The domain has no framework dependencies. FastAPI dependencies supply request-scoped SQLAlchemy sessions; future application services own transaction commits. Lifespan creates the engine/session factory and disposes the engine at shutdown. Connections are lazy. Settings can be injected directly for tests.

Provider and repository protocols establish adapter contracts. Concrete providers, persistence repositories and feature tables are deferred until use cases exist. Compare is a presentation mode, not a third execution engine. No LangGraph, generic CRUD abstraction, service locator or microservices are introduced.

`GET /api/health` returns `{"status":"ok","service":"researchlab-api"}`. It is API liveness only, not database/provider readiness. The shell displays a real health request outcome, never simulated AI results. Hash-based navigation supports reloads and browser history without a routing dependency.

Application logs are JSON with timestamps, levels, events, request status and duration. They omit prompts, bodies, URLs and credentials. `--no-access-log` disables Uvicorn's separate text access log. CORS allows explicit configured origins and GET only; expand methods with future endpoints.

Alembic enables pgvector explicitly; startup never mutates schemas. The migration role needs extension creation privileges. Downgrade deliberately retains the extension to preserve potentially shared vector data while rewinding the Alembic revision. PostgreSQL binds to loopback and persists in a named volume. `docker compose down` preserves data; adding `-v` deletes it. The Docker image is version tagged. Frontend dependencies use the committed npm lockfile; Python dependencies use bounded compatible ranges.

## Scope boundary

Authentication, ingestion, vector tables/indexes, inference, agent tools, evaluation, LangGraph and deployment are future work. Production hosting will need API routing and operational configuration. Review Milestone 1 before ingestion work begins.

## Milestone 1 verification (2026-10-21)

- Backend: 14 tests passed on Python 3.14.7. Installed Starlette emits one upstream deprecation warning about its httpx TestClient integration.
- Frontend: TypeScript check and Vite production build passed on Node 24.19.0.
- Alembic: offline upgrade SQL generated successfully, including extension creation.
- Docker Compose and live migrations: unverified because Docker is not installed in this environment.
- Browser visual/interaction checks: not performed; responsive styles are implemented, but browser review remains useful.

Framework references: [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/), [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/), [Vite builds](https://vite.dev/guide/build), [pgvector](https://github.com/pgvector/pgvector).
