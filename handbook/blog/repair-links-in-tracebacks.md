# Why put repair links in tracebacks?

A JAX traceback can tell you where a value became a tracer. The next step is understanding what the failing operation expected and how to confirm a repair. That step gets harder when the error comes from an accelerator, a compiled kernel, or a job that finished overnight.

JAX Diagnostics takes a cue from Solid 2.0. A diagnostic includes a stable ID and a link to a focused repair guide. The guide names the evidence to collect and the check that would demonstrate a fix. [Solid 2.0 diagnostics](https://www.solidjs.com/blog/solid-2-0-the-long-hold/)

## From traceback to fix

Consider a JIT-compiled function that calls `if flag` on an input. Inside tracing, Python needs a concrete boolean to choose a branch. JAX reports `TracerBoolConversionError`. With [the training wrapper installed](../quickstart.md), the traceback ends with a note pointing to [JAX-TRACE-003](../reference/tracing.md#jax-trace-003).

If the branch depends on the runtime value, one repair is `jax.lax.cond`. Test both branches under JIT. If the flag is truly configuration, static specialization can be appropriate, with a compilation cost for distinct values. The guide helps choose based on what the argument represents.

## One guide for a person and an agent

At a Python entry point, the wrapper attaches a note to the exception. For a traceback from a completed run, the CLI routes the saved text. An agent can load the same guide as a skill and use it to organize a reproduction. All three use the same diagnostic IDs.

The hard cases are ambiguous backend messages. An out-of-memory report can come from a large temporary, replication, or memory-space constraints. The current message router suggests a starting point. The [roadmap](../roadmap.md) prioritizes real failure fixtures and better evidence for decisions like that one.
