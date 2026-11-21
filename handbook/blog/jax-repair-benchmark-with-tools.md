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

Upstream docs had the lowest observed median time and gross input token count. The skill condition had the highest median wall time and output token count and ran ten public smoke tests, versus nine for the raw traceback and eight for upstream docs. The raw traceback already supported verified repairs across all four cases, including the implicit derivative. All three conditions tied on verified fixes in this first round. Gross and uncached token counts are both shown because repeated tool turns can reuse large parts of the prompt.

In the first round, the router missed two failures. The batched-padding case raised a slice-bounds `IndexError`, and the bounded solver raised a reverse-mode `ValueError`. The agent received only the skill index for those two cases. The numerical and tracer cases already received focused guide sections.

## Rerun with the missing routes

We added `JAX-INDEX-002` for dynamic slice bounds under JIT and `JAX-GRAD-002` for reverse-mode differentiation through dynamic loops. The slice guide distinguishes a moving window of fixed size from a runtime extent that needs static padding and masking. The loop guide keeps the runtime stopping rule while giving autodiff a static iteration bound. Both original tracebacks matched their new guides in preflight. [Slice bounds guide](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/shapes.md#jax-index-002) · [Dynamic loops guide](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/transforms.md#jax-grad-002)

We reran all 24 trials using the same model, tool access, schedule, and independent checks. Only the skill prompts for batched padding and the bounded solver changed. Every patch passed again. Each cell shows the first round → rerun. Time and token counts are medians, and smoke-test counts are totals.

| Context | Verified fixes | Time (s) | Gross input tokens | Output tokens | Uncached input tokens | Smoke tests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw traceback | 8/8 → 8/8 | 29.33 → 26.31 | 101,475 → 100,540 | 755.5 → 713.5 | 13,824 → 13,946.5 | 9 → 8 |
| Upstream docs | 8/8 → 8/8 | 24.86 → 27.06 | 91,204.5 → 103,835 | 687.5 → 655 | 14,144 → 14,659.5 | 8 → 8 |
| JAX Diagnostics | 8/8 → 8/8 | 38.55 → 38.69 | 105,943.5 → 134,036.5 | 1,141.5 → 1,090.5 | 14,620.5 → 15,343.5 | 10 → 11 |

Across just the four skill trials that received new guidance, median gross input fell 9.4% (118,173 → 107,025.5). Median uncached input rose from 14,620.5 to 15,121.5, output rose from 1,065 to 1,078.5, and time moved from 37.69 to 38.01 seconds. All four patches passed in both rounds. The bounded solver accounted for the lower gross input, and batched padding accounted for both extra smoke tests (two → four). [Rerun measurements](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/tool_repair/results-routed.json) · [Submitted patches](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/tool_repair/proposals-routed.json)

The new routes gave both failures specific guidance. This rerun showed no reduction in time, uncached input, or test runs on those cases. Even with identical prompts, the raw-traceback and upstream-docs medians moved by about three seconds and two seconds in opposite directions. We ran the old and updated skill in separate rounds, so these measurements cannot isolate the effect of the new guidance from run-to-run variation.

This is still a small CPU study. We ran two attempts per condition, tested public reports that the model may have encountered, and used bounded reproductions. Per-arm medians describe the observed runs. A repeatable difference across models, runtimes, or concurrency levels remains unverified. A follow-up with more held-out cases could add cases requiring multiple hypotheses, such as accelerator-only bugs and failures whose traceback identifies the wrong-looking layer.

## Side-by-side test of shorter guidance

We shortened the shared skill instructions and the four guide sections used here. Each still names the boundary the repair must preserve, such as static output shape for padding or the original stopping rule for a solver. The text supplied by the skill, including the routed excerpt, was 37–41% smaller across the four cases.

We ran both versions in the same round with the same Codex CLI, model, reasoning effort, CPU runtime, and independent verifier as above. Each of the four cases ran twice per version in fresh sessions. Four waves each started two matched pairs, with the old and shorter guidance represented in every wave. The paired agent prompts shared the same task, traceback, and route. Only the guidance text differed. [Benchmark source and instructions](https://github.com/mlnomadpy/jax-diagnostics/tree/main/benchmarks/tool_repair)

Four sessions timed out before returning a patch, and four ended with a Codex workspace routing error before any tool calls. These affected four complete pairs. We relaunched those pairs together with the same prompts and retained the original attempts in the per-trial record. The comparison uses the completed attempts and measures repair time, so the time spent waiting on the interrupted sessions is outside the comparison.

| Metric | Original guidance | Shorter guidance |
| --- | ---: | ---: |
| Verified fixes | 8/8 | 8/8 |
| Median agent time | 37.29 s | 36.68 s |
| Median gross input tokens | 106,176 | 105,650.5 |
| Median uncached input tokens | 14,584.5 | 14,300.5 |
| Median output tokens | 1,059 | 1,052.5 |
| Public smoke tests, total | 10 | 8 |

Shorter guidance used fewer gross input tokens in seven of eight matched pairs. The median within-pair saving was 27,271 gross input tokens, and total gross input fell 12.8% (954,081 → 832,206). The pooled medians in the table are closer because they rank the eight runs in each condition separately. For uncached input, the median within-pair saving was 551 tokens. Gross input counts cached tokens and repeated tool context, so the two token measures describe different parts of the workload.

The shorter version was faster in five of eight pairs, with a median within-pair saving of 1.09 seconds. Both extra smoke tests came from one original-guidance batched-padding run. Across these four cases, the shorter version kept all verified fixes and used less input overall. Two repetitions per case give limited evidence about how repeatable the timing and smoke-test differences are. [Per-trial measurements and interrupted attempts](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/tool_repair/results-paired-skills.json) · [Submitted patches](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/tool_repair/proposals-paired-skills.json)
