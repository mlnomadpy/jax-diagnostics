# Can a coding agent fix harder JAX bugs with tools?

Our [first repair benchmark](jax-repair-benchmark.md) gave a coding agent one attempt at each of four JAX failures and asked it to return code without using tools. All three context conditions passed every case. This follow-up lets the agent inspect files, edit code, and execute smoke tests. We also asked more of the patches. Making the immediate exception go away had to preserve derivatives, batching, and behavior on edge cases.

## Four cases taken from public reports

| Case | Failure mechanism | Independent verification |
| --- | --- | --- |
| Masked angles | Singular `atan2` inputs can contaminate gradients even when the forward result masks them | JIT and batching, mixed masks, all-inactive examples, and analytic gradients for coordinates and weights |
| Batched padding | Slicing by a runtime extent inside JIT requires static slice bounds | One through three spatial dimensions, zero and full extents, and gradients for the values and padding |
| Bounded solver | Reverse-mode autodiff rejects a dynamically stopped loop | States, iteration counts, early exit, zero steps, the hard cap, and analytic derivatives for three inputs |
| Closed-over implicit VJP | A custom derivative captures a traced parameter rather than receiving it explicitly | Batched finite-iteration forward values and a separate implicit VJP, including non-unit cotangents |

These are reduced application-level reproductions, with explicit contracts to make scoring possible. The reports motivated the failure mechanisms. The benchmark does not claim to repair the full original applications. [Masked gradient report](https://github.com/jax-ml/jax/issues/1052) · [Dynamic padding report](https://github.com/jax-ml/jax/discussions/22458) · [MJX reverse-mode report](https://github.com/google-deepmind/mujoco/issues/2259) · [JAXopt custom VJP report](https://github.com/google/jaxopt/issues/31)

## What each agent saw

We used Codex CLI 0.169.1 with `gpt-6.1-sol` and low reasoning effort. Each case ran twice per condition, for 24 fresh sessions. The runner shuffled trial order with a fixed seed and allowed four concurrent trials. Every session got the same behavior contract, broken module, real traceback, and public smoke test for its case.

- **Raw traceback** received the common material.
- **Upstream docs** also received a relevant docstring from the installed JAX release, capped at 2,400 characters.
- **JAX Diagnostics** also received `SKILL.md` and the matched guide section when the router found one.

Agents could inspect and edit their workspace and execute local commands. They were asked to use `./check` and run it after the repair. Independent tests ran in a separate directory after the agent exited. Before the trials, we confirmed that the broken code failed its smoke test and that a reference patch passed both the public and independent tests on CPU with JAX and jaxlib 0.10.0. [Source, methodology, and evidence](https://github.com/mlnomadpy/jax-diagnostics/tree/main/benchmarks/tool_repair)

## Results

**All 24 patches passed the independent checks.**

| Context | Verified fixes | Median time | Median input tokens | Median output tokens | Median uncached input | Smoke tests, total (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw traceback | 8/8 | 29.33 s | 101,475.0 | 755.5 | 13,824.0 | 9 (1) |
| Upstream docs | 8/8 | 24.86 s | 91,204.5 | 687.5 | 14,144.0 | 8 (1) |
| JAX Diagnostics | 8/8 | 38.55 s | 105,943.5 | 1,141.5 | 14,620.5 | 10 (1) |

Elapsed time covers the Codex CLI invocation, including tool calls and smoke tests. Independent verification is outside that timer. Input tokens are the counts reported by Codex, including harness and tool context and cached input. Cached input counts are retained separately. Test runs count invocations of the public `./check` wrapper. Each submitted patch was also evaluated once by the independent harness. [Per-trial measurements](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/tool_repair/results.json)

Upstream docs had the lowest observed median time and gross input token count. The skill condition had the highest median wall time and output token count and ran ten public smoke tests, versus nine for the raw traceback and eight for upstream docs. The raw traceback already supported verified repairs across all four cases, including the implicit derivative. These runs show no verified-fix advantage for the current skill. Gross and uncached token counts are both shown because repeated tool turns can reuse large parts of the prompt.

Both of the routing misses were visible before any agent ran. The batched-padding failure had a generic slice-bounds `IndexError`, and the bounded solver raised a reverse-mode `ValueError` before the router could assign a specific guide. The skill condition received its index for those cases. The numerical and tracer cases received focused sections. These are concrete places to improve the router and rerun the same checks.

This is still a small CPU study. We ran two attempts per condition, tested public reports that the model may have encountered, and used bounded reproductions. Per-arm medians describe the observed runs. A repeatable difference across models, runtimes, or concurrency levels remains unverified. A follow-up with more held-out cases could add cases requiring multiple hypotheses, such as accelerator-only bugs and failures whose traceback identifies the wrong-looking layer.
