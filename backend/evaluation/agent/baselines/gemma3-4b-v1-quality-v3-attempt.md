# ResearchLab agent evaluation - gemma3-4b-v1-quality-v3-attempt

Dataset: `researchlab-agent-eval-v1`; 1 questions x 1 repeats = 1 attempted runs.
Model: `ollama/gemma3:4b` at temperature 0; retrieval strategy `semantic`.
Hardware: `Intel64 Family 6 Model 181 Stepping 0, GenuineIntel`; 12 physical / 14 logical CPUs; 15.46 GiB visible RAM; 0.68 GiB available at report.
Scope prompt `scope-v2`; document schema `document-action-schema-v3`; grounding schema `grounded-answer-schema-v2`.

| Metric | Value | Denominator |
| --- | ---: | ---: |
| Tool-selection accuracy | 1.0 | 1 attempted runs |
| Document-scope compliance | 1.0 | 1 attempted runs |
| Expected-status accuracy | 0.0 | 1 attempted runs |
| Grounded-answer rate | 0.0 | 1 grounded-labelled runs |
| Abstention accuracy | 0.0 | 1 attempted runs |
| Unanswerable abstention rate | n/a | 0 abstention-labelled runs |
| Citation validity | n/a | 0 returned citations |
| Citation coverage | 0.0 | 1 expected references |

End-to-end latency: p50 `61347.63` ms; p95 `61347.63` ms.
Configured-timeout overruns: `0` attempts.
Completed runs: `1`; errors: `0`.
Tool calls: mean `2.0`, range `2..2`. Graph steps: mean `9.0`, range `9..9`.
Decision variability: 0 of 1 questions varied across repeats; stable-question rate `1.0`.

## Cases

| Question | Repeat | Expected | Actual | Decision path | Tool calls | Citations | Latency ms | Error |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| d03_low_battery_response | 1 | document_research/grounded | document_research/insufficient_evidence | search_documents[evidence_needed] -> search_documents[evidence_needed] -> insufficient_evidence[evidence_does_not_support_answer] | 2 | 0 | 61347.63 |  |

## Interpretation limits

Citation validity checks only expected document/page/chunk identity and verbatim quote containment. It does not measure factual entailment, semantic faithfulness or completeness.
This small synthetic dataset is suitable for reproducible regression signals, not claims about production-domain quality. Timing and local-model decisions depend on hardware and runtime state.

Baseline comparison: incompatible; regression=False.
