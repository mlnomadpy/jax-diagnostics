# JAX Diagnostics

A standalone agent skill that routes JAX and accelerator failures to focused repair guides. Inspired by [Solid 2.0 diagnostics](https://www.solidjs.com/blog/solid-2-0-the-long-hold/).

Start with the [diagnostic index](SKILL.md). Give your agent the exception, the failing transform or operation, and the backend. For example, a `TracerBoolConversionError` points to [JAX-TRACE-001](references/tracing.md#jax-trace-001) and a TPU-only NaN points to [JAX-NUM-001](references/numerics.md#jax-num-001).

To use it as a Codex skill, clone this repository into `~/.codex/skills/jax-diagnostics` and invoke `$jax-diagnostics` with your error or symptom. The skill is documentation and works without installing JAX, Flax, or Python. Existing JAX errors and application code can be used as-is.

Each guide gives a trigger, a way to distinguish likely causes, and a check for the repair. The diagnostic IDs belong to this repository.
