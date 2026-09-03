# Extraction provenance

This standalone project was extracted on 2026-09-03 from the
`top1_intension_retrieval` branch of the LLMGen repository at commit `83d0d27`.

Copied and then isolated:

- controlled synthesis primitives;
- the generation orchestrator and audit lifecycle;
- v2 candidate, synthesis, decision-policy, and LabelDesc assets;
- synthesis, protocol, circuit-breaker, axis, and v2 contract tests.

Intentional extraction changes:

- package imports use `top1_data_gen` and no longer depend on LLMGen;
- the CLI defaults directly to the v2 assets;
- only the minimal JSONL and dataset validation contract is retained;
- credential loading requires HTTPS;
- manifest implementation hashes refer to the standalone package sources.

No API key, model output, raw response, cache, rejected attempt, run directory,
checkpoint, training implementation, or historical pilot was copied.
