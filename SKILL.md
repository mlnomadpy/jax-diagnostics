---
name: jax-diagnostics
description: Route JAX exceptions and symptoms from tracing, numerical failures, recompiles, accelerator memory, Orbax restores, or cross-framework parity to a focused debugging guide.
---

# JAX diagnostics

Select a route from the final exception and the first operation that fails. If only a symptom is available, treat the route as a hypothesis. Record the failing transform, input shapes and dtypes, package versions and backend before changing the reproducer. Follow the verification step for the selected route.

| Evidence | Guide |
|---|---|
| `ConcretizationTypeError`, `TracerBoolConversionError`, `TracerIntegerConversionError`, `TracerArrayConversionError`, `NonConcreteBooleanIndexError` | [JAX-TRACE-001](references/tracing.md#jax-trace-001) |
| `UnexpectedTracerError` or a tracer reported by the leak checker | [JAX-TRACE-002](references/tracing.md#jax-trace-002) |
| NaN, Inf, non-finite gradients, or finite eager output with non-finite compiled output | [JAX-NUM-001](references/numerics.md#jax-num-001) |
| JAX and another framework disagree on the same checkpoint | [JAX-PARITY-001](references/numerics.md#jax-parity-001) |
| Repeated JIT compilation or slow first execution | [JAX-COMPILE-001](references/runtime.md#jax-compile-001) |
| Device OOM, unexpected replication, or TPU kernel memory failure | [JAX-MEM-001](references/runtime.md#jax-mem-001) |
| Orbax restore fails or continuation diverges after restore | [JAX-RESTORE-001](references/runtime.md#jax-restore-001) |

Include the diagnostic ID and observed evidence in the handoff. When no route fits, preserve the exact error and narrow the failing operation first. Use existing reproduction artifacts before requesting a new accelerator run.
