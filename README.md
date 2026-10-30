# ResearchLab

A portfolio AI engineering workspace with independently testable RAG and Agentic modes. **Milestone 3C-A adds a reproducible, rollback-isolated evaluation baseline for the bounded LangGraph document-research agent** alongside PDF ingestion, local retrieval, optional reranking and grounded RAG answers. The Agentic mode can choose zero or more document searches and returns validated citations for grounded answers. Compare remains unimplemented. The default local workflow makes no paid API calls.

## Requirements

Python 3.12+, Node.js 22.12+ with npm, Docker Compose v2, and Ollama for local answer generation. Python 3.14 and Node 24 are available in the implementation environment. The local PostgreSQL connection uses port 5433; Docker CLI availability depends on your host.

## Local development

From the repository root (PowerShell):

```powershell
Copy-Item .env.example .env
```

Set your own nonempty `POSTGRES_PASSWORD` in `.env`. Set `DATABASE_URL` to `postgresql+psycopg://<user>:<URL-encoded-password>@localhost:5433/<database>`, matching the PostgreSQL variables. URL-encode reserved characters in the URL password, keeping the raw password in `POSTGRES_PASSWORD`. Match the URL port to `POSTGRES_PORT`. No credential defaults are provided; never commit `.env`.

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

Expect a `vector` row and revision `0004_keyword_search`. For offline migration SQL, set `DATABASE_URL` and run `alembic -c backend/alembic.ini upgrade head --sql` using the virtual environment executable. Offline generation does not verify a live database.

## Architecture

```text
backend/
  app/
    api/              HTTP routes and request-scoped session dependency
    application/      Ingestion, retrieval, RAG and bounded LangGraph orchestration/services
    domain/           Framework-independent document, RAG and Agentic contracts
    evaluation/       Offline RAG and Agentic datasets, metrics, runners and CLIs
    infrastructure/   PDF, local models, separate RAG/Agent Ollama adapters and PostgreSQL
    main.py           Application factory and dependency composition
  migrations/         Alembic environment and versioned migration
  tests/              Unit/API, PostgreSQL, semantic and opt-in real-provider tests
frontend/
  src/                RAG and Agentic workspaces, real API states and execution trace display
compose.yaml          PostgreSQL 17 + pgvector with persistent volume
```

This pragmatic modular monolith keeps operations simple and defines seams for independent future RAG and Agentic tests. The domain has no framework dependencies. FastAPI dependencies supply request-scoped SQLAlchemy sessions; future application services own transaction commits. Lifespan creates the engine/session factory and disposes the engine at shutdown. Connections are lazy. Settings can be injected directly for tests.

Provider and repository protocols establish adapter contracts. Ingestion and retrieval share the configured embedding adapter/profile. Agentic decisions/direct answers use a separate structured provider contract, while RAG and Agentic grounded answers share one bounded evidence-assembly and citation-validation service. The document tool wraps `RetrievalService`; it cannot bypass embedding-profile checks, document filters or strategy behavior. Compare is a presentation mode, not a third execution engine. No generic CRUD abstraction, service locator or microservices are introduced.

`GET /api/health` returns `{"status":"ok","service":"researchlab-api"}`. It is API liveness only, not database/provider readiness. The shell displays a real health request outcome, never simulated AI results. Hash-based navigation supports reloads and browser history without a routing dependency.

Application logs are JSON with timestamps, levels, events, request status and duration. They omit prompts, bodies, URLs and credentials. `--no-access-log` disables Uvicorn's separate text access log. CORS allows explicit configured origins and GET/POST for health, document upload/listing, retrieval and questions.

Alembic enables pgvector explicitly; startup never mutates schemas. The migration role needs extension creation privileges. Downgrade deliberately retains the extension to preserve potentially shared vector data while rewinding the Alembic revision. PostgreSQL binds to loopback and persists in a named volume. `docker compose down` preserves data; adding `-v` deletes it. The Docker image is version tagged. Frontend dependencies use the committed npm lockfile; Python dependencies use bounded compatible ranges.

## Scope boundary

Authentication, ANN indexes, web/external tools, memory persistence, multi-agent orchestration, evaluation dashboards and deployment remain outside Milestone 3C-A. Production hosting needs API routing and operational configuration. Review the measured baseline before changing routing or prompts.

## Milestone 1 verification (2026-10-21)

- Backend: 14 tests passed on Python 3.14.7. Installed Starlette emits one upstream deprecation warning about its httpx TestClient integration.
- Frontend: TypeScript check and Vite production build passed on Node 24.19.0.
- Alembic: offline upgrade SQL generated successfully, including extension creation.
- Docker Compose and live migrations: unverified because Docker is not installed in this environment.
- Browser visual/interaction checks: not performed; responsive styles are implemented, but browser review remains useful.

Framework references: [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/), [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/), [Vite builds](https://vite.dev/guide/build), [pgvector](https://github.com/pgvector/pgvector).

## PDF ingestion (Milestone 2A)

Use the RAG screen's PDF form or POST multipart field `file` to `/api/documents`. A successful request returns HTTP 201 with `id`, sanitized `filename`, `page_count`, `chunk_count`, `embedding_model`, `embedding_revision`, `embedding_profile` and `embedding_dimension`. The upload form displays real loading, success and error states. Successful uploads refresh the document selection in the RAG question panel below it.

```powershell
curl.exe -F "file=@C:/path/to/document.pdf;type=application/pdf" http://127.0.0.1:8000/api/documents
```

The default upload limit is 10 MiB. Both the `.pdf` filename suffix and `application/pdf` MIME type are required, followed by PDF signature and parser checks. Filenames are metadata only and are never used as paths. An ASGI receive wrapper bounds the whole multipart body to the file limit plus 64 KiB overhead, including requests without Content-Length, before disk spooling. The endpoint independently checks the actual file size. Originals are not retained: PostgreSQL stores metadata, extracted chunk text and vectors only.

Text extraction uses pypdf in a separate process. The parent monitors a 15-second deadline and 256 MiB resident-memory budget, terminating the parser when exceeded. Limits include worker startup and are sampled, not an OS sandbox or a strict allocation ceiling. No PDF JavaScript, attachments or external tools are executed. Parsing does not perform OCR. Encrypted, malformed and textless PDFs are rejected. Pages may be blank when other pages contain text. Defaults cap documents at 200 pages, 2 million extracted characters and 5,000 chunks. Limits are environment settings in `.env.example`. This is a local development endpoint, without authentication, rate limiting or a production file-scanning sandbox.

Chunks use character windows (default 1,000 characters, 150-character overlap) independently within each page. They never cross pages. Page numbers are one-based; `chunk_order` is zero-based across the entire document. Whitespace is trimmed at page boundaries, NUL characters are removed, and blank chunks are omitted. Each chunk stores the document UUID, page number, order, text and a 384-dimensional pgvector vector. Documents retain the file hash, counts, model identity and chunking parameters for reproducibility.

The provider interface remains synchronous and provider-agnostic. Ingestion uses `embed_documents`; retrieval uses `embed_query` on the same configured provider and verifies its `profile` against the corpus before comparing vectors. Both paths delegate to one shared encoder: NFKC Unicode normalization, NUL removal, whitespace collapse, pinned MiniLM tokenizer/mean pooling, 256-wordpiece limit, float32 output and L2 normalization. No query/document-specific prompts are added. Chunk text is retained as extracted; this normalization is applied only for embedding.

The default provider is `sentence_transformers`, model `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with 384 dimensions. It uses CPU inference and safetensors, never remote custom model code. The revision is an immutable Hugging Face commit, not a moving branch. Configuration selects the adapter/model, cache, batch size, CPU threads and offline behavior; unsupported provider/model pairs fail validation. The legacy `local` / `local-hash-v1` adapter remains available explicitly for lexical corpora and tests, with no silent fallback from semantic inference.

`embedding_profile` is a SHA-256 identity of the model name/revision, dimension, preprocessing/pooling contract, sequence limit and normalization. Documents store the identity and readable model revision. A singleton `embedding_corpus` binding and a foreign key prevent committing documents from different profiles into the corpus. Ingestion checks compatibility before extraction and again during the write transaction; concurrent first uploads serialize through the singleton key. The upload returns 409 `embedding_profile_mismatch` on a mismatch, preserving all existing rows. The same dimension alone never implies model compatibility.

### Local semantic model setup

The first MiniLM use downloads public model files; input text never leaves the process. No paid service, account or API key is required. Install CPU PyTorch explicitly to avoid CUDA packages on systems whose default PyTorch index selects a GPU build:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e "./backend[dev]"
.\.venv\Scripts\python.exe -m app.infrastructure.embeddings
```

The last command preloads the configured model. Weights and tokenizer files live under the ignored `.cache/models` by default; relative configured cache paths resolve against the repository root. Once downloaded, set `EMBEDDING_LOCAL_FILES_ONLY=true` to enforce cache-only operation. A missing cache or failed model load yields upload HTTP 503 without saving a partial document. Health remains API liveness only. Use `.env.example`'s semantic model settings; a previous `.env` with `local/local-hash-v1` must be updated deliberately. Only model-selection keys in the implementation workspace's `.env` were updated; credentials were not changed or displayed.

The pinned safetensors weights are 90,868,376 bytes (about 91 MB / 86.7 MiB), plus roughly 0.7 MB of tokenizer assets and small configuration files. Only the selected PyTorch format is needed; downloading the entire Hub repository also fetches alternative formats unnecessarily. Dependencies are separate: this Windows Python 3.14 CPU PyTorch wheel was about 126 MB, with additional scientific-library downloads and a substantially larger installed footprint. Sizes vary by platform/version. References: [pinned model files](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), [CPU PyTorch installation](https://pytorch.org/get-started/locally/).

CPU execution needs no GPU. Defaults use 2 CPU threads and batches of 16. A recent multi-core CPU with several GB of free RAM is a practical development starting point, not a measured minimum. The semantic dependency bounds keep Sentence Transformers 5.1 and Transformers 4.57 compatible; upgrades should rerun the semantic tests. The model loads lazily once per process/configuration and serializes encoding through a lock to limit oversubscription. Multiple Uvicorn workers multiply model RAM; batch size and sequence length affect transient memory and throughput. First use includes model download/import/load latency, while subsequent requests reuse the model. Large PDFs still run synchronous ingestion in the worker pool. MiniLM truncates after 256 wordpieces; character-based chunks can exceed that limit, particularly code or unusual text. This milestone does not add tokenizer-aware chunking or long-document pooling. The model is primarily English and is not a guarantee of retrieval quality. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [encoding API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

### Existing lexical data and re-embedding

Migration `0003_embedding_profiles` does not delete or recompute vectors. Known `local-hash-v1` rows are backfilled with the exact legacy fingerprint; an unknown or already mixed legacy corpus gets an incompatible quarantine fingerprint and retains its original model labels/text/vectors. An empty database remains unbound until its first successful ingestion. Switching configuration alone never relabels old vectors as semantic.

To convert an existing corpus to the configured model, stop or pause uploads, preload/cache the target model, then explicitly run:

```powershell
.\.venv\Scripts\python.exe -m app.infrastructure.embeddings
.\.venv\Scripts\python.exe -m app.infrastructure.reembed --apply
```

This maintenance command uses the same provider interface and document encoding path. It re-embeds stored chunk text, preserving document/chunk IDs, filenames, text, hashes, page/order metadata and creation timestamps. The whole conversion is one PostgreSQL transaction. It locks corpus/document/chunk tables against writes while reads remain available, updates every vector/model revision/profile together, and rolls everything back if any model/validation/database step fails. The deferred compatibility FK is checked at commit. Concurrent ingestion finishing after conversion rechecks compatibility and cannot commit an old profile. An interruption with uncertain commit status can be checked by inspecting corpus/documents and rerunning; documents already at the target profile are skipped. The tool requires `--apply`; ingestion never converts data automatically.

Changing model revision, tokenizer/preprocessing/pooling contract or normalization requires re-embedding and a new fingerprint. Changing dimensions additionally requires a schema/settings migration; `vector(384)` is fixed in this milestone. Re-embedding reuses existing chunks and does not change chunk size/overlap; re-chunking is a separate future migration. Large-corpus conversion can hold locks for a long time and generate substantial PostgreSQL WAL; a future staged conversion may be appropriate at that scale. No database/volume reset is needed.

Exact-byte SHA-256 deduplication returns HTTP 409, usually with the existing document ID (a concurrent unique-constraint race may omit it). Semantically identical PDFs with different bytes are distinct. Extraction and embedding finish before the write transaction. The document and all chunks are inserted atomically; any flush or commit failure rolls back both. A second hash check plus PostgreSQL's unique constraint handles concurrent uploads. Original filenames/text/credentials are omitted from application error logs.

Error responses use `detail: {code, message}` for ingestion errors: 413 `file_too_large`, 415 `unsupported_type`, 422 `empty_file` / `invalid_pdf` / `pdf_limit_exceeded`, 409 `duplicate_document` / `embedding_profile_mismatch`, and 503 `embedding_failed` / `storage_unavailable`. Missing multipart fields use FastAPI's normal 422 validation response. An unconfigured database returns 503. The UI does not automatically retry an ambiguous network failure; a retry may return 409 if the earlier request committed.

Migration `0002_document_ingestion` adds `documents` and `chunks` without changing existing data. It includes unique file hashes, unique per-document chunk order, positive metadata/count checks, a fixed dimension constraint, cascade foreign key and indexes on creation time and document/page. The document/order unique index also serves ordered per-document scans. ANN indexes are deferred until retrieval establishes its metric and workload. Apply with the existing `alembic upgrade head` command. Never reset the database or delete volumes to migrate.

### PostgreSQL integration tests

First apply migrations to the existing local PostgreSQL database, then run:

```powershell
$env:RUN_DB_TESTS='1'
.\.venv\Scripts\python.exe -m pytest backend/tests
Remove-Item Env:RUN_DB_TESTS
```

Without `RUN_DB_TESTS=1`, PostgreSQL tests skip explicitly; unit/API validation tests still run. Integration tests use the configured `.env` database with a private schema per test, an outer transaction and nested savepoints. They create ORM tables in that private schema and copy only the public Alembic revision; outer rollback removes the schema and test data. Public document/chunk/corpus rows are never rewritten or locked by these tests. Run `alembic check` separately against the real application schema to verify migration fidelity. They do not reset schemas, truncate tables or delete volumes. They validate real PDF upload/persistence, pgvector width, page/order/overlap metadata, duplicates, database constraint errors and failures after partial writes. PostgreSQL sequences can advance during rolled-back tests.

Parser reference: [pypdf text extraction and memory considerations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html). Persistence reference: [pgvector SQLAlchemy integration](https://github.com/pgvector/pgvector-python#sqlalchemy).

### Earlier Milestone 2A verification, before semantic improvements (2026-10-27)

- PostgreSQL on port 5433: additive migration applied to `0002_document_ingestion`; Alembic schema-drift check passed. pgvector 0.8.1 is enabled, and chunk embeddings use `vector(384)` with the expected indexes.
- Backend: 48 tests passed, including six live PostgreSQL integration tests. Two focused provider/upload tests passed again after the final factory consistency change. Test writes rolled back; final document/chunk counts remained zero.
- Frontend: typecheck and production build passed. Browser visual/interaction verification was not performed.
- A Windows ownership problem in old pytest temporary/cache directories required `-p no:cacheprovider --basetemp=.venv/pytest-m2a-20261027-a`. For a fresh run, choose a new temporary directory under the ignored `.venv`; pytest may remove an existing basetemp directory, so never point it at source or valuable data. No application workaround was added for these host permissions.
- One upstream Starlette/httpx TestClient deprecation warning remains. Local embeddings are lexical development vectors; OCR and semantic embedding models are not included.

The earlier Milestone 2A verification did not include retrieval or question answering; those are added in Milestone 2B below.

### Semantic tests and browser verification

Semantic tests use the genuine cached MiniLM weights and compare paraphrased queries with relevant and unrelated passages. They assert relative ranking/separation, not exact cosine scores, and check that query/document preprocessing produces the same vector. Enable both test groups:

```powershell
$env:RUN_DB_TESTS='1'
$env:RUN_SEMANTIC_TESTS='1'
.\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider --basetemp=.venv/pytest-choose-a-new-unused-name
Remove-Item Env:RUN_DB_TESTS, Env:RUN_SEMANTIC_TESTS
.\.venv\Scripts\alembic.exe -c backend/alembic.ini current
.\.venv\Scripts\alembic.exe -c backend/alembic.ini check
```

The earlier browser automation runtime was unavailable. The reviewer subsequently confirmed the Milestone 2A PDF upload and duplicate-upload checks before that commit. Those checks do not verify the new question flow; follow the Milestone 2B manual steps below.

### Historical Milestone 2A verification and dependency recovery (2026-10-27)

The semantic runtime is installed in the existing `.venv` using the original selected versions: Sentence Transformers 5.1.2, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.5.4, SciPy 1.18.1 and scikit-learn 1.9.1. Required regex, joblib, threadpoolctl, narwhals and cloudpickle dependencies are installed too. `pip check` reports no broken requirements. Previously installed package versions were preserved.

After recovery, the five previously failing tests passed, followed by all 61 backend tests with PostgreSQL and semantic groups enabled. Genuine normalized 384-dimensional MiniLM embeddings passed paraphrase-ranking and document/query preprocessing checks; real PDF upload/persistence passed too. The pinned weights were loaded offline from the existing cache, with unchanged SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`; they were not downloaded again. The historical recovery reports are under ignored `.cache/`.

The final review additionally isolates PostgreSQL tests from any manual-upload data using private transaction-scoped schemas. Final review passes all 62 backend tests with PostgreSQL and semantic tests enabled, including the public-corpus isolation regression. Frontend typecheck/build and `pip check` pass; `alembic check` reports no schema drift at revision `0003_embedding_profiles`. Git review found no credential values or unintended generated artifacts in the candidate files; `.env`, virtual environments, model downloads, caches and local reports remain ignored. The existing upstream Starlette/httpx TestClient deprecation warning remains. No browser check is claimed from automated tests: the manual PDF and duplicate-upload checks must be confirmed by the reviewer before commit/push.

Milestone 2A did not add retrieval, question answering, LangGraph or deployment. The following section describes the current Milestone 2B behavior.

## Semantic retrieval and grounded QA (Milestone 2B)

### Provider setup

The development default uses the native Ollama `/api/chat` API at `http://127.0.0.1:11434` with `gemma3:4b`. Ollama and this model were already available in the implementation environment, so no new model download or API key was needed. Start Ollama if its desktop app/service is not already serving, then verify the model:

```powershell
ollama serve
# In another terminal:
ollama list
```

If `gemma3:4b` is absent, explicitly download it with `ollama pull gemma3:4b` before asking questions. This can consume several GB of disk space, network bandwidth and setup time; the application never pulls a model automatically. Select another installed model through `LLM_MODEL` if desired and rerun the smoke/manual checks. A smaller model may be faster and less reliable at following the evidence format. Unsupported provider names fail settings validation; adding another provider requires an adapter/factory entry behind the existing LLM interface, not changes to RAG orchestration.

The new defaults in `.env.example` configure top-k, similarity threshold, context/prompt budgets, provider/model, timeout and response limits. Existing MiniLM weights are reused; retrieval must match the ingestion model revision, preprocessing and 384-dimensional profile. If the corpus contains lexical or otherwise incompatible vectors, use the existing explicit re-embedding workflow rather than resetting the database.

Local Ollama inference has no hosted API bill, but uses CPU/GPU, RAM, electricity and time. The 4B generator has a substantially larger memory footprint than MiniLM; model loading, hardware and prompt length affect latency. There is no automatic remote fallback. If you change `LLM_BASE_URL` to a remote Ollama-compatible service, question text and selected context are sent there, and that service's privacy, access and cost policies apply. The current adapter does not implement API-key authentication or hosted OpenAI/Anthropic adapters. Never place credentials in a URL or commit them.

### API and retrieval behavior

| Endpoint | Request / result |
| --- | --- |
| `GET /api/documents?limit=100&offset=0` | Paginated metadata: `documents`, `total`, `limit`, `offset`. Limit is 1-100; the UI supports loading more. |
| `GET /api/rag/config` | Effective default/max top-k, similarity threshold, LLM provider/model and maximum question characters. This is configuration, not provider readiness. |
| `POST /api/retrieval` | JSON `{ "query": "your question", "top_k": 5, "document_ids": ["<UUID>"] }`; returns matching chunks with document/filename/page/order, text, cosine distance/similarity, embedding profile and retrieval latency. |
| `POST /api/rag/questions` | JSON `{ "question": "your question", "top_k": 5, "document_ids": ["<UUID>"] }`; returns an answer or explicit insufficient-evidence status, citations, counts, latencies and supported token usage. |

Both POST bodies accept omitted/null `top_k` to use `RETRIEVAL_TOP_K` (default 5), or an integer from 1-20. Omitted/null `document_ids` searches all documents; `[]` selects none and never broadens to all. At most 100 UUIDs are accepted. Unknown IDs return no matches for those IDs. Questions must be nonempty, at most 2,000 characters and 4,000 UTF-8 bytes, without unsupported control characters. JSON request bodies are capped at 32 KiB. The local development app has no authentication/tenant authorization; document filtering is an evidence scope, not an access-control boundary.

Queries use the same MiniLM encoding path and model profile as ingestion. PostgreSQL computes exact cosine distance with pgvector `<=>`; smaller distance ranks first, with deterministic tie-breaking. `RETRIEVAL_MIN_SIMILARITY` defaults to 0.25 and filters weak matches before top-k. It is a configurable retrieval cutoff, not a calibrated confidence score. Document filters and the expected profile are applied in the SQL query. The repository checks the corpus profile before embedding and rechecks it under a shared row lock in the search transaction, so a concurrent corpus conversion cannot mix vector spaces. The read transaction finishes before generation.

There is no new schema migration for Milestone 2B: the existing `vector(384)` storage supports the query. Alembic remains at `0003_embedding_profiles`. Exact scans are pragmatic for the current local corpus; they scan/rank eligible vectors and will become slower as the library grows. ANN indexes, hybrid search and reranking are intentionally deferred.

### Grounding and bounded generation

RAG validates the question, embeds it, retrieves eligible chunks and assembles a bounded source prompt. Defaults allow 4,000 UTF-8 bytes of source text and 6,000 bytes for the combined system/question/source prompt. Chunks may be truncated at UTF-8 boundaries to fit. Ollama receives an 8,192-token context setting and a 512-token output cap; settings reserve a further 1,024-token margin and reject incompatible byte/token budget combinations. Byte bounds are conservative controls, not exact model-tokenizer counts.

The model receives source labels and text as untrusted data, with instructions to ignore embedded commands, use no outside knowledge, and acknowledge insufficient context. Structured generation uses a strict discriminated schema: `status=answered` requires a nonempty answer and evidence list, while `status=insufficient_evidence` permits neither field. The service validates source labels against only the chunks actually sent and verifies each quote by NFKC/whitespace-normalized, case-sensitive substring matching against its supplied source excerpt. Whitespace may be normalized, but capitalization and punctuation are not silently repaired. Document UUID, filename, page number, chunk ID/order and supporting text come from database records, never from model-invented metadata. `[S1]` labels are local to that response; the UI links them to the displayed source details.

An answered response has `status="answered"`, nonempty `answer` and citations containing `source_id`, `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_order`, `quote` and the bounded supporting `text`. When retrieval is empty, generation is skipped. Empty retrieval or a model-declared inability returns `status="insufficient_evidence"`, `answer=null`, an explanatory `message` and no citations. A provider response that claims an answer without supporting citations is rejected with 502. Unsupported labels, unverifiable quotes or malformed/truncated provider output are rejected rather than replaced with a fabricated answer.

These checks establish source identity and quote presence; they do not prove that every generated claim logically follows from its quote or make the model immune to prompt injection. Review source passages before relying on an answer. There is no reranker, entailment verifier or evaluation dashboard in this milestone.

Validation failures return 422, oversized requests 413 and incompatible profiles 409. Embedding/storage/provider unavailability returns 503; provider timeout returns 504; invalid, oversized, compressed, truncated or unsupported provider responses return 502. The shared Ollama transport wraps the complete HTTP exchange in an absolute client-side wall deadline: `LLM_TIMEOUT_SECONDS` defaults to 120 seconds, each graph call receives the smaller remaining execution budget, connection setup is capped at five seconds, and responses are capped by `LLM_MAX_RESPONSE_BYTES` (default 65,536). It disables environment proxies and redirects. `LLM_CONNECT_RETRIES` defaults to one and applies only to a failed TCP connection before a response; read/write timeouts, HTTP errors and completed inference are not transport-retried. A parsed grounded response that fails strict evidence validation may be regenerated once with an exact-copy instruction, within the original deadline. There is no permissive parsing, fabricated citation, model fallback or silent provider switch.

The absolute deadline cancels ResearchLab's async HTTP coroutine and closes the client connection; it does not prove that the separate Ollama process stopped model computation. Ollama may continue work after a disconnect, and Python cannot forcibly terminate that server-side operation. A process or machine suspended by the operating system can also observe the deadline only after it is scheduled again. Database calls and local embedding calls remain synchronous boundaries and cannot be forcibly interrupted once inside their native driver/runtime.

Responses and JSON logs expose retrieval/generation latency, retrieved/context chunk counts and prompt/completion token usage when Ollama supplies it. Missing usage remains null. Logs contain event names, timings, counts and outcomes; they do not log questions, document text, quotes, filenames, API keys or provider response bodies. Answers are returned to the caller without storing a conversation history.

### Tests and real-provider smoke check

Ordinary unit/API and PostgreSQL RAG tests mock LLM calls. PostgreSQL tests retain the private-schema/rollback isolation described above, and semantic tests use the existing MiniLM cache. Run the full automated groups from the root:

```powershell
$env:RUN_DB_TESTS='1'
$env:RUN_SEMANTIC_TESTS='1'
$env:EMBEDDING_LOCAL_FILES_ONLY='true'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider --basetemp=.venv/pytest-choose-a-new-unused-name
.\.venv\Scripts\alembic.exe -c backend/alembic.ini current
.\.venv\Scripts\alembic.exe -c backend/alembic.ini check
```

The real-provider test is separate and opt-in. It sends a small synthetic evidence prompt to the configured provider, so it requires Ollama/model availability and consumes real generation resources:

```powershell
$env:RUN_LLM_SMOKE='1'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_llm_smoke.py -q -p no:cacheprovider
Remove-Item Env:RUN_LLM_SMOKE
```

A real `gemma3:4b` adapter smoke check passed in the implementation environment: 28.895 seconds, 134 prompt tokens and 53 completion tokens. This is one setup check, not a throughput benchmark or browser verification. No paid service was used.

### Manual end-to-end verification

The reviewer confirmed that all three Milestone 2B manual browser checks passed on 2026-10-28. This confirmation is separate from automated API/model/database tests; browser automation was not used. With migrations current, MiniLM cached, Ollama running and FastAPI/Vite started using the setup commands above:

1. Open http://localhost:5173/#rag. Upload a small, unencrypted PDF with selectable text containing a known test fact, such as "The fictional support desk closes at 17:00 UTC." Confirm upload succeeds and the document library refreshes.
2. Select only that PDF, ask "When does the support desk close?" and choose a small top-k. Confirm a real loading state, then an answer supported by the known fact. Open its `[S1]` source link and verify filename, page, document/chunk IDs and quote against the PDF. Inspect the Network response for actual timings/counts/token usage.
3. Upload an unrelated PDF and select only it. Repeat the support-desk question. Verify retrieval cannot cite the excluded support PDF; expect insufficient evidence when the selected passages do not support the answer. Clear selection and verify the UI requests a selection rather than silently searching all documents. The API's explicit `document_ids: []` must also return no evidence.
4. Ask a question absent from all uploaded documents. Confirm the response acknowledges missing evidence and does not invent citations. Inspect `POST /api/retrieval` through http://127.0.0.1:8000/docs to check chunk ranking, filters and top-k independently of generation.
5. Upload the exact same PDF again, including after renaming it without changing bytes. Confirm the existing duplicate error and unchanged document count. Agentic and Compare screens must remain marked unfinished.
6. With Ollama unavailable or `LLM_MODEL` set to a nonexistent local model, submit a supported question and confirm a meaningful provider error, without a fabricated fallback answer. Restore the service/model afterward.
7. To check compatibility, point configuration at a different embedding profile without rewriting stored vectors and confirm a 409 mismatch for a nonempty search scope. Restore the original MiniLM configuration. Do not reset volumes, relabel old vectors or run conversion merely to bypass the guard.

LangGraph, hybrid search, reranking, evaluation dashboards and deployment remain outside this milestone.

### Milestone 2B verification (2026-10-28)

- All **170 backend tests passed** with `RUN_DB_TESTS=1`, `RUN_SEMANTIC_TESTS=1` and `RUN_LLM_SMOKE=1`, including PostgreSQL ranking/filter/profile/citation tests, genuine MiniLM inference, the real Ollama adapter and a full PDF-to-MiniLM-to-pgvector-to-RAG citation test. Ordinary LLM test cases remain mocked; real-provider tests require opt-in.
- Frontend typecheck and production build passed.
- `alembic upgrade head`, revision inspection and schema-drift checks passed at `0003_embedding_profiles`; no schema change was needed. `pip check` passed.
- The existing upstream Starlette/httpx TestClient deprecation warning remains. The reviewer confirmed all three manual browser checks passed; automated browser execution is not claimed. The manual steps above remain available for future verification.

At the Milestone 2B checkpoint, no deployment, Milestone 2C work, LangGraph, hybrid search, reranking or evaluation dashboard had been added.

## Reproducible RAG evaluation baseline (Milestone 2C-A)

Milestone 2C-A adds an offline evaluator around the existing `RetrievalService` and optional `RagService`; it does not add an API endpoint or frontend dashboard. The versioned dataset is under `backend/evaluation/datasets/v1/`. Its four short documents, questions and annotations were written for this repository and dedicated under CC0 1.0. They describe fictional habitat operations, include paraphrased questions, one question with two relevant chunks, a document-filtered question and two unanswerable questions. Each relevance annotation is an exact tuple of dataset document key, one-based page and zero-based global chunk order. The dataset's chunk size and overlap are pinned, and loading fails if an expected reference is not produced.

Evaluation data never enters the production corpus. Each run creates ORM tables and vectors in a randomly named PostgreSQL schema inside an outer transaction, searches that schema, then rolls the entire transaction back. The runner compares qualified public document/chunk counts before and after and fails if they changed. It does not reset a database, truncate tables or delete volumes. The existing MiniLM cache and pgvector extension are reused; retrieval-only evaluation makes no paid or network provider calls.

From the repository root, run the deterministic retrieval ordering and metrics offline:

```powershell
$env:EMBEDDING_LOCAL_FILES_ONLY='true'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Set-Location backend
..\.venv\Scripts\python.exe -m app.evaluation.cli `
  --run-id local-minilm-check `
  --k 1 3 5 `
  --output ..\.cache\evaluations\latest.json `
  --report ..\.cache\evaluations\latest.md `
  --fail-on-regression
Set-Location ..
Remove-Item Env:EMBEDDING_LOCAL_FILES_ONLY, Env:HF_HUB_OFFLINE, Env:TRANSFORMERS_OFFLINE
```

An editable/package install also exposes `researchlab-eval`; `python -m app.evaluation.cli` works directly from `backend` without reinstalling. JSON is the machine-readable artifact and Markdown is the concise report. Both local outputs default to ignored `.cache` paths. Results use sorted K values, dataset order, stable document UUIDs and deterministic pgvector tie-breaking. Run timestamps and measured latencies naturally vary. The default checked-in comparator is `backend/evaluation/baselines/minilm-v1.json`; `--fail-on-regression` returns exit code 2 for any retrieval metric decrease or an incompatible dataset, embedding profile or retrieval configuration.

Retrieval metrics are macro-averaged over the six answerable questions; the two unanswerable questions have no relevant-item denominator and are excluded from these four metrics:

- Recall@K is relevant expected references found in the first K divided by all expected references for that question.
- Precision@K is relevant expected references in the first K divided by K. If the similarity threshold returns fewer than K results, the missing positions count as nonrelevant.
- MRR@K is the reciprocal rank of the first relevant reference within K, or zero.
- Hit Rate@K is the fraction of answerable questions with at least one relevant reference within K.

Use `--with-llm --answer-top-k 5` to opt into the configured real local Ollama provider. This adds citation precision (dataset-relevant citations / all citations), citation coverage (unique expected references cited / expected references), normalized quote-presence validity and answer/abstention behavior. Provider failures are reported as errors and excluded from completed-case denominators; they are never converted into answers or abstentions. Generation p50/p95 and token totals are recorded only when returned by the provider. LLM results are informational and may vary by model/runtime; retrieval metrics are the regression gate.

These answer checks are deliberately mechanical. They establish whether a citation identifies an annotated source and whether its normalized quote occurs in the supplied chunk. They do not determine whether every claim is entailed by that quote, so the reports set `factual_entailment_measured=false` and do not claim faithfulness.

### Baseline results and interpretation

The checked-in MiniLM baseline produced Recall@1 0.750, Precision@1/MRR@1/Hit Rate@1 0.833, Recall@3 and Recall@5 1.000, and MRR@3/MRR@5 0.917. Retrieval latency on the reference Windows CPU run was p50 18.83 ms and p95 23.79 ms for eight queries; timing is hardware-dependent and is not a regression threshold.

The opt-in `ollama/gemma3:4b` reference run completed all eight questions at K=5. Citation precision and citation coverage were 0.714286, mechanical quote validity was 1.0, abstention behavior accuracy was 0.75, unanswerable abstention was 1.0 and answerable response rate was 0.666667. Generation latency was p50 18.51 s and p95 44.97 s. Ollama supplied usage for all cases: 3,094 prompt tokens and 828 completion tokens. Two answerable questions were conservatively refused, and one multi-source answer cited two irrelevant retrieved chunks; this is visible rather than hidden by an aggregate-only score.

This is a small baseline, not a statistically strong benchmark. Do not tune prompts, thresholds or questions against these eight examples and then present the same-set score as generalization. Preserve this version for comparisons; create a new dataset version for annotation changes, and keep a separate unseen holdout before optimizing. Do not upload these fixtures into a manual/production corpus during evaluation. The corpus is synthetic and cannot establish real-domain quality, prompt-injection resilience or factual entailment.

Milestone 2C-A added no migrations or frontend changes. Its semantic baseline remains the stable comparison point for Milestone 2C-B below.

## Hybrid retrieval and optional reranking (Milestone 2C-B)

Semantic retrieval remains the default. Both RAG POST endpoints now accept an optional `strategy` of `semantic`, `hybrid` or `hybrid_rerank`; omitting it uses `RETRIEVAL_STRATEGY`, whose default is `semantic`. The selected strategy is returned as `strategy` by `/api/retrieval` and `retrieval_strategy` by `/api/rag/questions`. Citation metadata, bounded context, document filters and embedding-profile checks are unchanged.

Migration `0004_keyword_search` adds a stored English `tsvector` generated from chunk text and a GIN index. Keyword search passes the validated question as a bound value to PostgreSQL `websearch_to_tsquery('english', ...)`, ranks matches with `ts_rank_cd`, and applies document/profile filters before limiting. Raw web-search punctuation cannot become SQL. English stemming is useful for the current corpus but is not multilingual tokenization; changing the text-search configuration requires a migration and a new evaluation.

Hybrid retrieval independently takes `SEMANTIC_CANDIDATE_COUNT` and `KEYWORD_CANDIDATE_COUNT` candidates, then applies Reciprocal Rank Fusion using `1 / (RRF_FUSION_CONSTANT + rank)`. Duplicate results contribute at most once per input ranking and are deduplicated by persistent chunk ID. Final chunks retain their actual cosine distance even when discovered by keyword search, preserving the existing API contract. Defaults use 20 candidates from each ranking and an RRF constant of 60.

`hybrid_rerank` passes up to `RERANKER_CANDIDATE_COUNT` fused candidates to the local `cross-encoder/ms-marco-MiniLM-L6-v2` model pinned at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. It uses the existing Sentence Transformers runtime, CPU inference and safetensors, loads lazily on first use, and has no hosted or paid API fallback. A missing, invalid or unavailable model returns a provider error rather than silently falling back to another strategy.

The required cached cross-encoder snapshot in the implementation environment contains about 91.8 MB of files. Model loading and inference require additional RAM; expect a few hundred MB of working memory depending on PyTorch and batch size. On Windows without symlink support, the Hugging Face cache may consume extra disk space. To download and verify the exact pinned revision once:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -c "from app.infrastructure.config import Settings; from app.infrastructure.reranker import create_reranker_provider; p=create_reranker_provider(Settings()); print(p.model, p.revision, p.score('Which passage is relevant?', ['A relevant passage.', 'Unrelated text.']))"
Set-Location ..
```

After it is cached, set `RERANKER_LOCAL_FILES_ONLY=true`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for reproducible offline runs. `GET /api/rag/config` reports effective strategies, candidate counts, fusion constant and pinned reranker identity without loading the model.

Run all retrieval strategies on the exact Milestone 2C-A dataset and judgments:

```powershell
$env:EMBEDDING_LOCAL_FILES_ONLY='true'
$env:RERANKER_LOCAL_FILES_ONLY='true'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Set-Location backend
..\.venv\Scripts\python.exe -m app.evaluation.cli `
  --run-id local-hybrid-comparison `
  --strategies semantic hybrid hybrid_rerank `
  --k 1 3 5 `
  --output ..\.cache\evaluations\hybrid-comparison.json `
  --report ..\.cache\evaluations\hybrid-comparison.md
Set-Location ..
Remove-Item Env:EMBEDDING_LOCAL_FILES_ONLY, Env:RERANKER_LOCAL_FILES_ONLY, Env:HF_HUB_OFFLINE, Env:TRANSFORMERS_OFFLINE
```

The JSON contains per-strategy configurations, model revisions, raw rankings, metrics, p50/p95 latency and deltas versus semantic. Retrieval comparison does not call Ollama. `--with-llm --answer-strategy hybrid_rerank` remains separately opt-in for answer evaluation. The measured reference summary is versioned at `backend/evaluation/baselines/hybrid-comparison-v1.json`; the original semantic baseline remains the default regression gate.

The full backend verification enables `RUN_DB_TESTS=1`, `RUN_SEMANTIC_TESTS=1`, `RUN_RERANKER_TESTS=1` and, when Ollama is available, `RUN_LLM_SMOKE=1`. Set both local-files-only flags and the Hugging Face/Transformers offline flags after the two pinned models are cached; ordinary unit tests never download or invoke either model.

### Measured comparison

The real local comparison used the unchanged eight-question CC0 dataset, pinned MiniLM, exact pgvector cosine search, PostgreSQL full-text search and the pinned cached cross-encoder. Production counts remained 1 document / 3 chunks before and after the rollback-isolated run.

| Strategy | Recall@1 | Precision@1 | MRR@1 | Hit Rate@1 | Recall@3 | MRR@3 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic | 0.750 | 0.833 | 0.833 | 0.833 | 1.000 | 0.917 | 17.03 ms | 21.57 ms |
| hybrid | 0.750 | 0.833 | 0.833 | 0.833 | 1.000 | 0.917 | 20.87 ms | 23.27 ms |
| hybrid_rerank | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 82.12 ms | 133.51 ms |

At K=5 all strategies had Recall 1.000, Precision 0.233 and Hit Rate 1.000; reranking retained MRR 1.000 versus 0.917 for the other strategies. Plain hybrid produced no quality improvement on this dataset and added 3.84 ms p50 latency. Reranking improved first-result ordering but added 65.09 ms p50 and 111.94 ms p95 over semantic. This is evidence for keeping semantic as the default: the observed reranker gain is promising, but eight synthetic questions are too small to establish general improvement, and the latency trade-off is material. No claim is made about production-domain quality, multilingual search or factual answer faithfulness.

Milestone 2C-B adds no frontend dashboard, LangGraph, deployment or paid services. ANN search, more languages and larger held-out evaluation sets remain future work.

## LangGraph Agentic foundation (Milestone 3A, historical)

The Agentic tab now calls `POST /api/agent/questions`. A typed LangGraph `StateGraph` executes exactly one deterministic route and one configured LLM generation:

```text
START -> route -> direct_response ------------------------> finalize -> END
                 document_tools_unavailable_response ---->
```

The conservative router sends explicit requests about uploaded/selected PDFs or files to the limitation-aware branch; other questions use the direct branch. Both branches call the real Agentic generation adapter once. The document branch still generates a useful response, but its system instructions require an explicit limitation and prohibit claims of file inspection, grounding, quotes or citations. This is orchestration, not a document-research agent. It has no tools, browsing, conversation memory, checkpointer, retry/fallback loop or multi-agent behavior.

LangGraph is pinned to `1.2.18`. Only its in-process graph API is used: ResearchLab does not configure LangSmith tracing, a remote graph service or persistent graph state. Graph orchestration lives in `application/agent_graph.py`; request validation and result assembly live in `application/agent.py`; the provider protocol is in `application/ports.py`; native Ollama transport is isolated in `infrastructure/agent_llm.py`. The existing RAG adapter and its citation schema are unchanged.

`GET /api/agent/config` reports the effective provider/model and public execution limits. `POST /api/agent/questions` accepts `question` and optional `include_metadata` (default true). Responses always set `document_tools_available=false` and include a non-grounding disclaimer. Metadata contains the chosen route, total duration, node names/durations/outcomes and token counts when Ollama returns them. Setting `include_metadata=false` omits metadata from the response but does not disable safe internal timing. Logs include only event names, route, node, outcome, timing and step count; questions and model responses are not logged.

The graph defaults to eight maximum steps although a successful run uses three nodes. The adapter makes one request with a 120-second timeout, 384-token output limit and 65,536-byte response limit. These are execution bounds, not cancellation of model computation after a client disconnect. The synchronous FastAPI endpoint relies on its worker thread while local generation runs. There is no server-wide concurrency limit or authentication yet, so production exposure is out of scope.

The Agentic adapter uses the same `LLM_BASE_URL`, `LLM_MODEL` and `LLM_CONTEXT_TOKENS` as RAG, plus the `AGENT_*` settings in `.env.example`. It never pulls a model, calls a paid service, accepts credentials in the URL or silently substitutes a mock/fallback. Start Ollama and explicitly install the configured model as described in Milestone 2B.

Run deterministic tests (all external generation mocked):

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_graph.py backend/tests/test_agent_llm.py backend/tests/test_agent_api.py
```

The opt-in smoke test executes the compiled graph against the configured real Ollama model and consumes local compute:

```powershell
$env:RUN_AGENT_SMOKE='1'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_smoke.py -m agent_smoke -s
Remove-Item Env:RUN_AGENT_SMOKE
```

### Manual Agentic verification

1. Start Ollama with the configured model, then FastAPI and Vite using the setup commands above.
2. Open http://localhost:5173/#agentic and confirm the UI clearly states that document tools, citations and memory are unavailable.
3. Ask `In one sentence, what is a vector embedding?`; confirm a real answer appears with route `direct` and the three-node trace.
4. Ask `Summarize my uploaded PDF`; confirm route `document_tools_unavailable` and that the response does not claim it read the file or provide citations.
5. Stop Ollama (or temporarily select a nonexistent model), submit again and confirm a provider error appears with no fabricated response. Restore the configuration afterward.
6. Return to the RAG tab and repeat one known grounded question to confirm upload/retrieval/citation behavior is unchanged.

Automated browser interaction was not performed for Milestone 3A. The frontend typecheck/build and API contract tests verify compilation and response handling; the visual and real-model flows above remain manual review items. Milestone 3A adds no database migration and does not change production documents or evaluation data.

## Tool-using document research agent (Milestone 3B)

Milestone 3B replaces the 3A keyword router with a real structured LLM decision at every decision node. The compiled typed graph is:

```text
START -> decide -- answer_direct ------> direct_response --------> finalize -> END
                 -- search_documents --> document_search -> observe -> decide
                 -- answer_grounded ---> grounded_response ------> finalize -> END
                 -- insufficient ------> insufficient_evidence --> finalize -> END
```

The decision adapter first makes a strict, scope-only LLM decision: ordinary definitions and conceptual questions default to `general_knowledge`, while `document_research` requires an explicit request for uploaded/provided files, sources or citations. A general decision maps only to `answer_direct`. A document decision uses a separate discriminated action schema containing `search_documents`, `answer_grounded` and `insufficient_evidence`; only the search branch exposes required `query` and `top_k` fields. Invalid or extra fields are rejected rather than silently coerced. This separation prevents constrained local models from treating required tool fields or empty observations as evidence that every question needs search. The model receives bounded observations from prior searches, not a keyword-derived route. Once any document search has run, an ungrounded direct answer is rejected: the execution must produce a citation-validated grounded answer or explicitly report insufficient evidence.

`document_search` is the only tool. Its narrow interface delegates to the existing `RetrievalService`, preserving semantic/hybrid/reranked strategy behavior, MiniLM profile compatibility, top-k validation and document filtering. Production composition creates the retrieval service lazily only when the tool is called, so a direct answer does not require a database query. The graph never calls SQLAlchemy repositories or embedding providers directly.

Search calls are bounded by `AGENT_MAX_TOOL_CALLS` (default 3), and each model-selected `top_k` is capped by `AGENT_TOOL_TOP_K` (default 5). Repeated canonical queries are blocked regardless of `top_k`; case, spacing, Unicode presentation and punctuation differences do not create a new search. Evidence is deduplicated by persistent chunk ID and capped at `AGENT_MAX_EVIDENCE_CHUNKS` (default 12). Decision observations and final source context have independent UTF-8 byte bounds. The graph also has a 20-step recursion limit and computes a 120-second total deadline by default; each provider request receives the remaining duration, and the shared HTTP transport now enforces that duration across connection, response-header wait and body read. This is a client deadline, not forcible cancellation of Ollama computation or a synchronous database/embedding operation.

Grounded generation uses the same `GroundedAnswerService` as the RAG application service. It bounds prompts, treats retrieved text as untrusted evidence, permits only supplied source labels, verifies each quoted substring against the supplied chunk and maps citations to the actual chunk ID, document UUID, filename, page and chunk order. Model-authored filenames/pages are never accepted. Empty or missing document results produce an explicit insufficient-evidence response; they are not converted into a general answer.

`POST /api/agent/questions` accepts:

```json
{
  "question": "What interval does the uploaded manual specify?",
  "document_ids": null,
  "strategy": "semantic",
  "include_metadata": true
}
```

`document_ids=null` permits the tool to search the whole corpus; a UUID list restricts every tool call to that set, while an empty list searches nothing. `strategy` may be `semantic`, `hybrid` or `hybrid_rerank`; omission uses the configured semantic default. Responses distinguish `direct`, `grounded` and `insufficient_evidence`. Grounded results include validated citations. Optional metadata includes node timings, safe tool query/activity summaries, tool-call count and aggregated provider usage when every provider call reports it. Logs contain node/tool names, timings, counts, route and outcome, but never questions, search queries, document content, filenames, quotes or provider response bodies.

Errors remain structured: invalid request questions return 422; model-generated invalid tool arguments or invalid structured output return 502; embedding/storage/reranker/provider availability errors return 503; embedding-profile mismatch returns 409; and provider or total-execution timeouts return 504. The graph never retries routing/tool decisions, silently changes retrieval strategy, substitutes a mock response or falls back from a failed grounded answer to an ungrounded one. Only the bounded transport connection retry and one strict grounded-validation regeneration described above are permitted.

Run deterministic Agentic tests without external inference:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  backend/tests/test_agent_graph.py `
  backend/tests/test_agent_llm.py `
  backend/tests/test_agent_api.py
```

Run rollback-isolated PostgreSQL tool tests after migrations are current:

```powershell
$env:RUN_DB_TESTS='1'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_db.py
Remove-Item Env:RUN_DB_TESTS
```

Opt into both real local Ollama paths only when the configured model is running. The grounded smoke test uses synthetic in-memory evidence and does not mutate PostgreSQL:

```powershell
$env:RUN_AGENT_SMOKE='1'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_smoke.py -m agent_smoke -s
Remove-Item Env:RUN_AGENT_SMOKE
```

### Manual Milestone 3B verification

1. Start PostgreSQL, apply migrations, start Ollama with the configured model, then run FastAPI and Vite.
2. Open http://localhost:5173/#agentic. Confirm document scope and retrieval strategy controls appear, and the UI states that web tools and memory remain unavailable.
3. Ask `What is an LLM?`. Confirm a `direct` response, zero document tool calls, no citations and a general-answer disclaimer.
4. Select a known uploaded PDF and ask a question that requires it. Confirm at least one document-search activity, a `grounded` response and citations matching the PDF filename, page, chunk and verbatim supporting text.
5. Ask about a fact absent from the selected document. Confirm an explicit insufficient-evidence response with no fabricated citation or claim of verification.
6. Submit a question using each configured retrieval strategy and confirm the selected strategy remains within the chosen document scope.
7. Stop Ollama and retry; confirm a structured provider error with no fallback answer. Restore Ollama afterward.
8. Return to the RAG tab and repeat a known grounded question/upload workflow to verify unchanged RAG behavior.

Milestone 3B adds no database migration and no new hosted/paid provider. Multiple LLM decisions plus final generation can be materially slower than RAG or the 3A direct path on CPU, and all calls share the total execution deadline. The model may choose suboptimal search queries or stop conservatively; hard tool/context limits prioritize safety and reproducibility over open-ended autonomy. No browser automation result is claimed until the manual checks above are performed.

## Reproducible Agentic evaluation baseline (Milestone 3C-A)

Milestone 3C-A evaluates the unchanged Milestone 3B `AgentService` and compiled LangGraph. It adds no API route, frontend dashboard, migration or production routing override. The CC0 dataset under `backend/evaluation/agent/datasets/v1/` contains three fictional documents and ten explicitly labelled questions: general knowledge, explicit document research, unanswerable document requests, paraphrases, ambiguous requests and a request that attempts to induce a repeated empty search. Labels define expected scope, terminal status, tool use, document filters and exact supporting document/page/chunk references.

The evaluator reuses the RAG evaluator's PostgreSQL isolation manager. Every run seeds deterministic document IDs into a randomly named transaction-scoped schema, executes the real agent against that schema, rolls the transaction back and compares qualified `public.documents`/`public.chunks` counts before and after. It never uploads fixtures to the production corpus, truncates tables, resets a database or deletes volumes.

Run deterministic metrics, failure-injection and graph-boundary tests without Ollama:

```powershell
$agentEvalCache = New-Item -ItemType Directory -Force .cache
$agentEvalTemp = Join-Path $agentEvalCache.FullName ("pytest-agent-eval-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -m pytest `
  backend/tests/test_agent_evaluation.py `
  backend/tests/test_agent_graph.py `
  backend/tests/test_agent_llm.py `
  --basetemp=$agentEvalTemp -p no:cacheprovider
```

Run the rollback-isolation test against PostgreSQL on port 5433:

```powershell
$env:RUN_DB_TESTS='1'
$agentEvalCache = New-Item -ItemType Directory -Force .cache
$agentEvalDbTemp = Join-Path $agentEvalCache.FullName ("pytest-agent-eval-db-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -m pytest backend/tests/test_agent_evaluation_db.py `
  --basetemp=$agentEvalDbTemp -p no:cacheprovider
Remove-Item Env:RUN_DB_TESTS
```

The failure matrix is deliberately split at existing boundaries: `test_agent_evaluation.py` verifies that malformed-provider, timeout and execution-limit failures remain visible and score as failures; `test_agent_llm.py` rejects malformed strict structured output; and `test_agent_graph.py` exercises empty retrieval, repeated calls with changed `top_k`, provider timeouts and graph-limit exhaustion. These tests use deterministic fakes and are separate from measured Ollama variability.

Run the configured real Ollama model from `backend`. `--repeats` accepts 1-20 and measures variability using the documented `scope|status|tool_used|tool_call_count|error_type` signature. Two repeats are the checked-in baseline configuration and can take many minutes on CPU:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m app.evaluation.agent_cli `
  --repeats 2 `
  --strategy semantic `
  --run-id local-agent-check `
  --output ..\.cache\evaluations\agent-latest.json `
  --report ..\.cache\evaluations\agent-latest.md `
  --fail-on-regression
Set-Location ..
```

For a CPU-conscious investigation, repeat `--question-id` to run only named cases. The JSON records the selected IDs and total dataset size, and the result is intentionally incompatible with the full baseline comparator:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m app.evaluation.agent_cli `
  --question-id d01_seedling_protection `
  --question-id d02_seedling_paraphrase `
  --repeats 1 `
  --output ..\.cache\evaluations\agent-reliability.json `
  --report ..\.cache\evaluations\agent-reliability.md
Set-Location ..
```

An editable install also exposes `researchlab-agent-eval`. The default comparator is `backend/evaluation/agent/baselines/gemma3-4b-v1.json`; comparison requires the same dataset checksum, selected question set, repeat count, provider/model, scope/document/grounding prompt and schema versions, retrieval strategy and embedding profile. Accuracy decreases are flagged, but local latency is recorded rather than used as a regression gate. JSON contains all labelled attempts, errors, safe tool activity, node outcomes, model/prompt/schema identity, token totals when available, production counts and a local CPU/memory snapshot. Markdown is a concise review report. Default local outputs are under ignored `.cache/evaluations/`.

Metric denominators are explicit in both formats. Tool-selection accuracy, document-scope compliance, expected-status accuracy and abstention accuracy use every attempted run; provider or execution errors count as incorrect instead of disappearing. Grounded-answer rate uses only runs labelled `grounded`, and unanswerable abstention rate uses only runs labelled `insufficient_evidence`. Citation validity divides returned citations that match an expected document/page/chunk and contain a verbatim source quote by all returned citations. Citation coverage divides matched expected references by all expected references across grounded-labelled runs. Tool calls and graph steps describe completed runs. End-to-end latency p50/p95 uses nearest-rank percentiles over all attempts.

The preserved pre-fix `ollama/gemma3:4b` baseline used temperature 0, semantic retrieval, MiniLM and 10 questions x 2 repeats. Tool-selection and scope-compliance accuracy were 0.85; expected-status and abstention accuracy were 0.75; grounded-answer rate was 0.375; unanswerable abstention was 1.0. The three returned citations had 1.0 mechanical validity/reference precision, while citation coverage was 0.375 over eight expected references. End-to-end latency was p50 19.44 s and p95 40.65 s. One question varied across repeats. Three attempts errored: one anomalous `AgentLLMTimeout` after approximately 1,203.65 s despite the configured 120-second limit, and two strict grounded-response `LLMInvalidResponse` failures. These failures remain unchanged in `gemma3-4b-v1.json` as the before record.

The separate post-fix reliability run `gemma3-4b-v1-reliability-after.json` intentionally evaluated only `d01_seedling_protection` and `d02_seedling_paraphrase`, once each, on the same versioned dataset and isolated PostgreSQL infrastructure. Both completed as grounded responses with valid expected citations; error count and configured-timeout-overrun count were zero. Latencies were 43.80 s and 46.92 s (two-sample p50/p95 43.80/46.92 s). Production counts remained `1 document / 3 chunks` before and after. This two-case run verifies the historical failure paths; it is not a replacement full baseline, a general quality improvement claim or a statistically meaningful latency comparison. Its comparator is correctly marked incompatible because the question selection and grounding prompt/schema versions changed. The run used CPU inference on an Intel Core Ultra 7 255U (12 cores/14 logical processors) with 15.5 GiB visible RAM and only about 1.5 GiB free at the measurement point; memory pressure and other local load materially limit latency interpretation.

The observed timeout root cause was transport semantics, not LangGraph ignoring its deadline: the graph propagated its remaining budget, but sync `httpx` timeouts bounded individual inactive I/O operations, and the manual elapsed check ran only after response bytes arrived. The shared async transport now wraps the entire exchange in an absolute cancellable client deadline. A targeted real `gemma3:4b` request configured for 1.2 seconds returned `AgentLLMTimeout` in 1.248 seconds on this machine. This demonstrates client-side enforcement, not guaranteed cancellation of Ollama's separate model computation.

Both historical grounded failures were reproduced with actual Gemma output. The JSON was structurally valid, but Gemma changed the source sentence `The shutters...` to lowercase `the shutters...`; the case-sensitive evidence check correctly rejected it. The new discriminated answer schema exposes answered/insufficient cross-field requirements to constrained generation, the prompt explicitly requires character-for-character evidence, and one strict regeneration is allowed within the same deadline. No case-insensitive match, fuzzy quote repair or fabricated citation was added.

Mechanical citation validity is not semantic entailment, factual correctness, completeness or faithfulness; `factual_entailment_measured` is always false. Ten synthetic questions are too small for production-quality claims, and CPU timing is hardware/runtime dependent. Do not tune routing, prompts or retrieval against this same dataset and report the result as generalization. Preserve v1 for comparisons, create a new dataset version for label/content changes, and maintain a separate unseen holdout before optimization. Real runs use local compute and may be slow, but do not require a paid API.
