# ResearchLab RAG evaluation baseline

Dataset `researchlab-rag-eval-v1` contains four CC0 synthetic documents, six answerable questions and two unanswerable questions. The baseline used the pinned 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` revision and exact pgvector cosine search with a 0.15 minimum similarity.

| K | Recall@K | Precision@K | MRR@K | Hit Rate@K |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.750 | 0.833 | 0.833 | 0.833 |
| 3 | 1.000 | 0.389 | 0.917 | 1.000 |
| 5 | 1.000 | 0.233 | 0.917 | 1.000 |

Retrieval latency on the reference Windows CPU run was p50 18.83 ms and p95 23.79 ms across eight queries. Timing is informational and hardware-dependent. Production counts were 1 document / 3 chunks before and after the private-schema run.

An optional `ollama/gemma3:4b` run at K=5 completed all eight cases. Citation precision and coverage were both 0.714286, normalized quote validity was 1.0, abstention behavior accuracy was 0.75, unanswerable abstention was 1.0 and answerable response rate was 0.666667. Generation latency was p50 18.51 s / p95 44.97 s; Ollama reported 3,094 prompt and 828 completion tokens. These generator results are informational and nondeterministic.

Citation scoring verifies dataset source identity and normalized quote presence. It does **not** establish factual entailment or faithfulness of every generated claim.
