---
name: jax-diagnostics
description: Route JAX exceptions and symptoms from tracing, numerical failures, recompiles, accelerator memory, Orbax restores, or cross-framework parity to a focused debugging guide.
---

# JAX diagnostics

Route the final exception with `python3 scripts/diagnose.py 'ErrorName'` or pipe the traceback to it. Treat symptom matches as hypotheses. If no route fits, isolate the first failing operation.

| Evidence | Guide |
| --- | --- |
| Tracing, conversion, boolean indexing | [Tracing](references/tracing.md) |
| Shapes, dtypes, indexing | [Shapes](references/shapes.md) |
| Jaxpr, gradients, axes, PRNG, Flax | [Transforms](references/transforms.md) |
| NaNs, Inf, framework mismatch | [Numerics](references/numerics.md) |
| Backend, memory, recompiles, sharding, Pallas, collectives, transfers, donation, Orbax | [Runtime](references/runtime.md) |

Record the transform, shapes, dtypes, versions, and backend. Reuse existing reproductions and verify under the original transform. Report the route ID and evidence.
