---
name: jax-diagnostics
description: Route JAX exceptions and symptoms from tracing, numerical failures, recompiles, accelerator memory, Orbax restores, or cross-framework parity to a focused debugging guide.
---

# JAX diagnostics

Select a route from the final exception and the first operation that fails. If only a symptom is available, treat the route as a hypothesis. Record the failing transform, input shapes and dtypes, package versions and backend before changing the reproducer. Follow the verification step for the selected route.

Run `python3 scripts/diagnose.py 'ErrorName'` from this skill directory. For a traceback, pipe the saved text into the same command. The script reads the supplied text locally and prints a diagnostic ID and guide link. Use the index below when a shell is unavailable.

| Evidence | Guide |
|---|---|
| Named JAX exception from tracing, conversion, or masking | [Tracing](references/tracing.md) |
| Symbolic dimension, shape, dtype or indexing exception | [Shapes, types, and indexing](references/shapes.md) |
| Jaxpr, PRNG reuse, gradients, mapped axes, or Flax state | [Transformations and state](references/transforms.md) |
| Non-finite values or JAX–PyTorch output mismatch | [Numerics](references/numerics.md) |
| Backend crashes, memory, recompilation, sharding, Pallas, collectives, transfers, donation, or Orbax | [Runtime and restore](references/runtime.md) |

An exact exception match identifies the error family. A symptom match suggests where to begin. Include the diagnostic ID and observed evidence in the handoff. When no route fits, narrow the failing operation before guessing. Reuse existing reproduction artifacts before requesting accelerator work.
