# ResearchLab agent evaluation - milestone-3ca-gemma3-4b-v1

Dataset: `researchlab-agent-eval-v1`; 10 questions × 2 repeats = 20 attempted runs.
Model: `ollama/gemma3:4b` at temperature 0; retrieval strategy `semantic`.
Scope prompt `scope-v2`; document schema `document-action-schema-v2`.

| Metric | Value | Denominator |
| --- | ---: | ---: |
| Tool-selection accuracy | 0.85 | 20 attempted runs |
| Document-scope compliance | 0.85 | 20 attempted runs |
| Expected-status accuracy | 0.75 | 20 attempted runs |
| Grounded-answer rate | 0.375 | 8 grounded-labelled runs |
| Abstention accuracy | 0.75 | 20 attempted runs |
| Unanswerable abstention rate | 1.0 | 6 abstention-labelled runs |
| Citation validity | 1.0 | 3 returned citations |
| Citation coverage | 0.375 | 8 expected references |

End-to-end latency: p50 `19443.13` ms; p95 `40652.15` ms.
Configured-timeout overruns: `1` attempts.
Completed runs: `17`; errors: `3`.
Tool calls: mean `0.647`, range `0..1`. Graph steps: mean `4.941`, range `3..6`.
Decision variability: 1 of 10 questions varied across repeats; stable-question rate `0.9`.

## Cases

| Question | Repeat | Expected | Actual | Tool calls | Citations | Latency ms | Error |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| g01_llm_definition | 1 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 28721.4 |  |
| g01_llm_definition | 2 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 9471.04 |  |
| g02_llm_paraphrase | 1 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 15044.96 |  |
| g02_llm_paraphrase | 2 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 6864.09 |  |
| g03_ambiguous_calibration | 1 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 15582.93 |  |
| g03_ambiguous_calibration | 2 | general_knowledge/direct | general_knowledge/direct | 0 | 0 | 7276.65 |  |
| d01_seedling_protection | 1 | document_research/grounded | None/error | n/a | 0 | 1203651.03 | AgentLLMTimeout |
| d01_seedling_protection | 2 | document_research/grounded | document_research/grounded | 1 | 1 | 38951.65 |  |
| d02_seedling_paraphrase | 1 | document_research/grounded | None/error | n/a | 0 | 40652.15 | LLMInvalidResponse |
| d02_seedling_paraphrase | 2 | document_research/grounded | None/error | n/a | 0 | 34551.88 | LLMInvalidResponse |
| d03_low_battery_response | 1 | document_research/grounded | document_research/insufficient_evidence | 1 | 0 | 22680.6 |  |
| d03_low_battery_response | 2 | document_research/grounded | document_research/insufficient_evidence | 1 | 0 | 19289.08 |  |
| d04_probe_schedule | 1 | document_research/grounded | document_research/grounded | 1 | 1 | 32638.43 |  |
| d04_probe_schedule | 2 | document_research/grounded | document_research/grounded | 1 | 1 | 24260.37 |  |
| u01_uniform_colour | 1 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 22670.16 |  |
| u01_uniform_colour | 2 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 19443.13 |  |
| u02_ambiguous_manual | 1 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 19107.35 |  |
| u02_ambiguous_manual | 2 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 14348.87 |  |
| r01_repeated_empty_search | 1 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 23033.96 |  |
| r01_repeated_empty_search | 2 | document_research/insufficient_evidence | document_research/insufficient_evidence | 1 | 0 | 17577.28 |  |

## Interpretation limits

Citation validity checks only expected document/page/chunk identity and verbatim quote containment. It does not measure factual entailment, semantic faithfulness or completeness.
This small synthetic dataset is suitable for reproducible regression signals, not claims about production-domain quality. Timing and local-model decisions depend on hardware and runtime state.
