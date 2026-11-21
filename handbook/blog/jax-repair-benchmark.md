# Can repair guides help a coding agent fix JAX bugs?

**All three conditions verified 4 of 4 fixes in this pilot.** With just the traceback, the agent solved every case in the set. Giving it official JAX docs or the JAX Diagnostics skill added context without a measured correctness improvement. With four cases and one attempt per condition, the timing and token differences are descriptive.

This is a first test of the idea behind [repair links in tracebacks](repair-links-in-tracebacks.md). We used public JAX reports, generated fresh tracebacks, and scored the returned code with independent checks.

Our [tool-enabled follow-up](jax-repair-benchmark-with-tools.md) runs two repeats per condition on four harder cases and counts test invocations.

## The cases

| Case | Reproduced failure | What the verifier checks |
| --- | --- | --- |
| Power iteration with early stopping | A traced comparison reaches a Python `if` | JIT with runtime iteration caps and tolerance, including zero steps and early stopping |
| Repeated matrix-vector products | NumPy tries to convert a traced array | Matrix order, empty sequences, and gradients under JIT |
| Scoring variable-length prefixes | A dynamic slice bound requires a static integer | Different widths, empty and full prefixes, and gradients |
| Vectorized random draws with zero keys | JAX 0.10.0 raises a spurious `KeyReuseError` | The empty batch works, nonempty draws are unchanged under JIT, and key-reuse checking stays enabled |

The first three are reduced versions of user-reported application failures. The fourth is an upstream JAX bug reproduced on the affected version. The cases preserve the failure mechanisms and add explicit behavior contracts for grading. [Convergence report](https://github.com/jax-ml/jax/discussions/19650) · [Matrix chain report](https://github.com/jax-ml/jax/discussions/5576) · [Dynamic slice report](https://github.com/jax-ml/jax/discussions/23087) · [Key reuse issue](https://github.com/jax-ml/jax/issues/37859)

## How we ran it

We used Codex CLI 0.169.1 with `gpt-6.1-sol` and low reasoning effort. Every case ran in a fresh session under each of three conditions:

1. **Raw traceback.** The buggy module, behavior requirements, reproduction command, and original traceback.
2. **Upstream docs.** The same material plus a relevant docstring from the installed JAX version, capped at 2,400 characters.
3. **JAX Diagnostics.** The same material plus the project's `SKILL.md` and the guide section selected by the router when an error matched.

All sessions returned a complete Python module without using tools. This isolates the effect of supplied context in one repair attempt. An evaluator ran the returned modules on local CPU using JAX and jaxlib 0.10.0, with test inputs the agent never received. The 12 trials ran in a seeded shuffled order with up to three concurrent requests. Elapsed time covers the Codex CLI invocation, including startup and model response. Verification runs are outside that timing. [Benchmark harness and evidence](https://github.com/mlnomadpy/jax-diagnostics/tree/main/benchmarks/repair)

## Results

| Context | Verified fixes | Median wall time | Median input tokens | Median output tokens |
| --- | ---: | ---: | ---: | ---: |
| Raw traceback | 4/4 | 8.15 s | 24,638 | 154 |
| Upstream docs | 4/4 | 8.31 s | 25,127.5 | 147.5 |
| JAX Diagnostics skill | 4/4 | 8.76 s | 25,123 | 166 |

These are CLI-reported counts. Input tokens include the agent harness and any cached input, so they are larger than the visible task prompts. One docs trial reported 12,160 cached input tokens. Timing also includes concurrent requests and service variability. The skill added roughly 485 median input tokens over raw traceback in this run. There was no observed reduction in elapsed time or token usage versus raw traceback. The [per-trial measurements](https://github.com/mlnomadpy/jax-diagnostics/blob/main/benchmarks/repair/results.json) include cache counts and schedule order.

**Verifier correction.** The first PRNG check demanded exact equality between JIT-compiled outputs and an eager reference. It flagged rounding differences even when the reference computation itself was compiled. We switched to matching the execution mode on both sides, kept the exact-equality requirement, and replayed all saved answers. All three PRNG fixes then passed. The model outputs and resource measurements were reused as recorded.

## What the pilot tells us

The raw traceback was already a strong baseline. For tracing failures it named the failing operation, often with useful advice from JAX. On the empty-batch defect, all three answers took the same approach. They special-cased an empty key batch and kept the original vectorized operation for nonempty batches. A key-reuse guide that assumed every such exception reflected incorrect caller code could easily have led to changing the PRNG sequence instead.

The variable-prefix example exposed a routing gap even though the agent repaired it with all three prompts. JAX 0.10.0 emitted a generic `IndexError` for a dynamic slice bound. The skill's router did not select a dedicated guide, so that condition received only the skill index. A focused fixture for this shape of error would make that gap visible to maintainers and provide something specific to improve.

This pilot has a ceiling effect and limited external validity. The reports are public and may be familiar from training. The examples were reduced, the execution was CPU-only, and the agent could not browse or iterate on failing tests. There is also no repeated-trial estimate of timing variance. For a stronger follow-up, we would sample harder failures, use more independent cases and multiple repeats, and compare full tool-using sessions. In particular, measure how often the agent identifies the right first experiment, how often the router abstains appropriately, and how many verification or accelerator runs it takes to reach a confirmed fix. Those outcomes are closer to the developer experience this project is trying to improve.
