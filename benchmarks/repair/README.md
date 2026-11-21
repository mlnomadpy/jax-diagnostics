# JAX repair pilot

This directory contains four reduced reproductions of public JAX reports, the prompts and verification logic, all 12 returned modules, and per-trial measurements. See the [published results](https://www.tahabouhsine.com/jax-diagnostics/docs/blog/jax-repair-benchmark/) for interpretation.

## Reproduce

The harness runs on macOS and uses `sandbox-exec` for reproducers and patch verification. It requires an authenticated Codex CLI supporting `gpt-6.1-sol`, Python 3.13, and the pinned CPU runtime below. A full run issues 12 model requests. `--check-only` verifies the original failures and prepares prompts without querying the agent.

```sh
uv venv --python 3.13 .venv-repair
uv pip install --python .venv-repair/bin/python \
  jax==0.10.0 jaxlib==0.10.0 numpy==2.4.6 scipy==1.18.1 \
  ml-dtypes==0.7.0 opt-einsum==3.4.0
JAX_PLATFORMS=cpu PYTHONPATH=. .venv-repair/bin/python benchmarks/repair/run.py /tmp/jax-repair-new-run
```

Use a fresh output directory. The runner records prompts, CLI JSON events, submitted modules, and measurements there. It measures the wall time of each CLI invocation. Token counts come from the CLI usage event and include harness context and any cached input. The three conditions share one prompt template and requirements, with added context from installed JAX docstrings or the checked-in skill. No condition gets the verification script.

## First run

- [Cases and source reports](cases.py)
- [Trial measurements](results.json)
- [Returned modules](proposals.json)

The first PRNG verifier compared JIT output with an eager reference and rejected all three empty-batch workarounds due to small rounding differences. The checked-in verifier compares both sides under JIT and keeps exact equality. The saved outputs were then checked against the corrected suite. The measurements include each original verifier status and the corrected result. Agent calls and timing were reused as recorded.
