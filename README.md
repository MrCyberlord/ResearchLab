# ResearchLab

A portfolio AI engineering workspace for RAG and future Agentic execution. **Milestone 2B: semantic retrieval and grounded question answering.** PDF upload, local MiniLM embeddings, document-filtered retrieval and cited answers are available. Agentic workflows and Compare results remain unimplemented. The default local workflow makes no paid API calls.

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

Expect a `vector` row and revision `0003_embedding_profiles`. For offline migration SQL, set `DATABASE_URL` and run `alembic -c backend/alembic.ini upgrade head --sql` using the virtual environment executable. Offline generation does not verify a live database.

## Architecture

```text
backend/
  app/
    api/              HTTP routes and request-scoped session dependency
    application/      Ingestion, retrieval and RAG services; adapter/repository protocols
    domain/           Framework-independent document, retrieval, answer and citation types
    infrastructure/   PDF worker, MiniLM, Ollama, PostgreSQL repositories and settings/logging
    main.py           Application factory and dependency composition
  migrations/         Alembic environment and versioned migration
  tests/              Unit/API, PostgreSQL, semantic and opt-in real-provider tests
frontend/
  src/                Upload, document selection, RAG questions, citations and API status
compose.yaml          PostgreSQL 17 + pgvector with persistent volume
```

This pragmatic modular monolith keeps operations simple and defines seams for independent future RAG and Agentic tests. The domain has no framework dependencies. FastAPI dependencies supply request-scoped SQLAlchemy sessions; future application services own transaction commits. Lifespan creates the engine/session factory and disposes the engine at shutdown. Connections are lazy. Settings can be injected directly for tests.

Provider and repository protocols establish adapter contracts. Ingestion and retrieval share the configured embedding adapter/profile. RAG uses a provider-agnostic LLM contract with a real Ollama adapter; Agentic execution remains deferred. Compare is a presentation mode, not a third execution engine. No LangGraph, generic CRUD abstraction, service locator or microservices are introduced.

`GET /api/health` returns `{"status":"ok","service":"researchlab-api"}`. It is API liveness only, not database/provider readiness. The shell displays a real health request outcome, never simulated AI results. Hash-based navigation supports reloads and browser history without a routing dependency.

Application logs are JSON with timestamps, levels, events, request status and duration. They omit prompts, bodies, URLs and credentials. `--no-access-log` disables Uvicorn's separate text access log. CORS allows explicit configured origins and GET/POST for health, document upload/listing, retrieval and questions.

Alembic enables pgvector explicitly; startup never mutates schemas. The migration role needs extension creation privileges. Downgrade deliberately retains the extension to preserve potentially shared vector data while rewinding the Alembic revision. PostgreSQL binds to loopback and persists in a named volume. `docker compose down` preserves data; adding `-v` deletes it. The Docker image is version tagged. Frontend dependencies use the committed npm lockfile; Python dependencies use bounded compatible ranges.

## Scope boundary

Authentication, hybrid search, reranking, ANN indexes, agent tools, evaluation dashboards, LangGraph and deployment remain outside this milestone. Production hosting needs API routing and operational configuration. Review Milestone 2B before expanding scope.

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

The model receives source labels and text as untrusted data, with instructions to ignore embedded commands, use no outside knowledge, and acknowledge insufficient context. Structured generation must return an answer, source references with supporting quotes and an `insufficient_evidence` flag. The service validates source labels against only the chunks actually sent and verifies each quote by NFKC/whitespace-normalized substring matching against its supplied source excerpt. Quote formatting does not need byte-for-byte equality; factual entailment is not mechanically verified. Document UUID, filename, page number, chunk ID/order and supporting text come from database records, never from model-invented metadata. `[S1]` labels are local to that response; the UI links them to the displayed source details.

An answered response has `status="answered"`, nonempty `answer` and citations containing `source_id`, `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_order`, `quote` and the bounded supporting `text`. When retrieval is empty, generation is skipped. Empty retrieval or a model-declared inability returns `status="insufficient_evidence"`, `answer=null`, an explanatory `message` and no citations. A provider response that claims an answer without supporting citations is rejected with 502. Unsupported labels, unverifiable quotes or malformed/truncated provider output are rejected rather than replaced with a fabricated answer.

These checks establish source identity and quote presence; they do not prove that every generated claim logically follows from its quote or make the model immune to prompt injection. Review source passages before relying on an answer. There is no reranker, entailment verifier or evaluation dashboard in this milestone.

Validation failures return 422, oversized requests 413 and incompatible profiles 409. Embedding/storage/provider unavailability returns 503; provider timeout returns 504; invalid, oversized, compressed, truncated or unsupported provider responses return 502. The Ollama adapter uses `LLM_TIMEOUT_SECONDS` (default 120), a five-second-or-smaller connect timeout, and `LLM_MAX_RESPONSE_BYTES` (default 65,536). It disables environment proxies and redirects. It does not retry generation or silently switch models/providers.

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

No deployment, Milestone 2C, LangGraph, hybrid search, reranking or evaluation dashboard was added.
