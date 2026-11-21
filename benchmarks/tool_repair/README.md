# Tool-enabled repair benchmark

This study uses four reduced reproductions from public JAX ecosystem reports. It compares the raw traceback, installed JAX documentation, and the JAX Diagnostics skill, with two fresh Codex sessions per case and condition. The corpus contains the original programs, one public smoke test per case, independent acceptance checks, and reference implementations used before the benchmark.

Run on macOS with the pinned CPU environment from the [first study](../repair/README.md). The runner uses `sandbox-exec` for reference checks and independent scoring. Each agent gets a fresh workspace with shell and edit tools. It receives the request, traceback, and assigned context. It can run the smoke test using `./check`. The independent verifier runs in a separate directory after the agent exits.

```sh
JAX_PLATFORMS=cpu PYTHONPATH=. \
  .venv-repair/bin/python benchmarks/tool_repair/run_with_tools.py /tmp/jax-tool-repair-new-run
```

Use a fresh output directory. Add `--preflight-only` to validate the cases and reference fixes without model calls. A full run submits 24 agent requests. The runner shuffles the schedule with seed 20261122 and allows four concurrent trials. Raw event logs, candidate modules, verification output, and measured rows are written to the output directory. Agent wall time includes tool calls and public checks but excludes the independent verifier. Test runs count calls to the public `./check` wrapper.

- [Case programs, provenance, and checks](cases.py)
- [Per-trial metrics](results.json)
- [Submitted patches](proposals.json)
