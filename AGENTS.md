# Repository Guidelines

## Scope

This repository only implements controlled Top1 intent-router data generation.
Keep generation, structured audits, resumability, manifests, and dataset validation
here. Do not add model training, inference serving, checkpoints, or experiment UI.

## Development

Use Python 3.12, four-space indentation, public type hints, and concise docstrings.
Run the following before committing:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m compileall -q src scripts tests
```

Commit subjects begin with `[data]`, `[code]`, or `[docs]`.

## Modeling and safety

Never implement routing, filtering, labeling, review, or fallback decisions with
keyword lists or regular expressions. Use explicit plan metadata and independent
model-based audits. Never relax strict schemas, candidate coverage, content-axis
coverage, prompt-quality gates, or audit consensus silently.

Never commit API keys, unreviewed model output, raw responses, request caches,
rejected attempts, or generated run directories. API endpoints must use HTTPS.
Reviewed versioned datasets may be committed only with provenance and a validation
summary.
