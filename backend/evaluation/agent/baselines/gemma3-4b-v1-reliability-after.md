# ResearchLab agent evaluation - gemma3-4b-v1-reliability-after

Dataset: `researchlab-agent-eval-v1`; 2 questions x 1 repeats = 2 attempted runs.
Model: `ollama/gemma3:4b` at temperature 0; retrieval strategy `semantic`.
Scope prompt `scope-v2`; document schema `document-action-schema-v2`.

| Metric | Value | Denominator |
| --- | ---: | ---: |
| Tool-selection accuracy | 1.0 | 2 attempted runs |
| Document-scope compliance | 1.0 | 2 attempted runs |
| Expected-status accuracy | 1.0 | 2 attempted runs |
| Grounded-answer rate | 1.0 | 2 grounded-labelled runs |
| Abstention accuracy | 1.0 | 2 attempted runs |
| Unanswerable abstention rate | n/a | 0 abstention-labelled runs |
| Citation validity | 1.0 | 2 returned citations |
| Citation coverage | 1.0 | 2 expected references |

End-to-end latency: p50 `43799.1` ms; p95 `46923.37` ms.
Configured-timeout overruns: `0` attempts.
Completed runs: `2`; errors: `0`.
Tool calls: mean `1.5`, range `1..2`. Graph steps: mean `7.5`, range `6..9`.
Decision variability: 0 of 2 questions varied across repeats; stable-question rate `1.0`.

## Cases

| Question | Repeat | Expected | Actual | Tool calls | Citations | Latency ms | Error |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| d01_seedling_protection | 1 | document_research/grounded | document_research/grounded | 1 | 1 | 43799.1 |  |
| d02_seedling_paraphrase | 1 | document_research/grounded | document_research/grounded | 2 | 1 | 46923.37 |  |

## Interpretation limits

Citation validity checks only expected document/page/chunk identity and verbatim quote containment. It does not measure factual entailment, semantic faithfulness or completeness.
This small synthetic dataset is suitable for reproducible regression signals, not claims about production-domain quality. Timing and local-model decisions depend on hardware and runtime state.
This run used CPU inference on an Intel Core Ultra 7 255U (12 physical/14 logical processors), with 15.5 GiB visible RAM and about 1.5 GiB available at reporting time.

Baseline comparison: incompatible; regression=False.
