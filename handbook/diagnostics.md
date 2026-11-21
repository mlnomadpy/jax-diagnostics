# Diagnostic catalog

These 12 exception routes cover the [pinned public JAX exception exports](https://github.com/jax-ml/jax/blob/7261b7bfe9c6e75d04a2e36f0387977def874fe0/jax/errors.py).

| Exception | Repair guide |
| --- | --- |
| `ConcretizationTypeError` | [JAX-TRACE-001](reference/tracing.md#jax-trace-001) |
| `UnexpectedTracerError` | [JAX-TRACE-002](reference/tracing.md#jax-trace-002) |
| `TracerBoolConversionError` | [JAX-TRACE-003](reference/tracing.md#jax-trace-003) |
| `TracerIntegerConversionError` | [JAX-TRACE-004](reference/tracing.md#jax-trace-004) |
| `TracerArrayConversionError` | [JAX-TRACE-005](reference/tracing.md#jax-trace-005) |
| `NonConcreteBooleanIndexError` | [JAX-TRACE-006](reference/tracing.md#jax-trace-006) |
| `InconclusiveDimensionOperation` | [JAX-SHAPE-001](reference/shapes.md#jax-shape-001) |
| `JaxprTypeError` | [JAX-TRANSFORM-001](reference/transforms.md#jax-transform-001) |
| `JAXTypeError` | [JAX-TYPE-001](reference/shapes.md#jax-type-001) |
| `JAXIndexError` | [JAX-INDEX-001](reference/shapes.md#jax-index-001) |
| `KeyReuseError` | [JAX-RNG-001](reference/transforms.md#jax-rng-001) |
| `JaxRuntimeError` | [JAX-RUNTIME-001](reference/runtime.md#jax-runtime-001) |

## Symptom routes

| Evidence | Repair guide |
| --- | --- |
| Pallas or Mosaic failure | [JAX-KERNEL-001](reference/runtime.md#jax-kernel-001) |
| Device memory exhaustion | [JAX-MEM-001](reference/runtime.md#jax-mem-001) |
| NaN or Inf | [JAX-NUM-001](reference/numerics.md#jax-num-001) |
| Orbax restore or resume mismatch | [JAX-RESTORE-001](reference/runtime.md#jax-restore-001) |
| Distributed initialization or collective failure | [JAX-DIST-001](reference/runtime.md#jax-dist-001) |
| Sharding or mesh mismatch | [JAX-SHARD-001](reference/runtime.md#jax-shard-001) |
| Disallowed transfer | [JAX-TRANSFER-001](reference/runtime.md#jax-transfer-001) |
| Donated or deleted buffer | [JAX-BUFFER-001](reference/runtime.md#jax-buffer-001) |
| Backend initialization | [JAX-BACKEND-001](reference/runtime.md#jax-backend-001) |
| Recompilation or slow tracing | [JAX-COMPILE-001](reference/runtime.md#jax-compile-001) |
| Flax state or scope mismatch | [JAX-STATE-001](reference/transforms.md#jax-state-001) |
| Cross-framework drift | [JAX-PARITY-001](reference/numerics.md#jax-parity-001) |
| Dynamic slice start, stop or step under JIT | [JAX-INDEX-002](reference/shapes.md#jax-index-002) |
| Reverse-mode differentiation through dynamic loops | [JAX-GRAD-002](reference/transforms.md#jax-grad-002) |
| Gradient or custom derivative | [JAX-GRAD-001](reference/transforms.md#jax-grad-001) |
| Mapped axes or scan carry | [JAX-AXIS-001](reference/transforms.md#jax-axis-001) |
| Shapes or broadcasting | [JAX-SHAPE-002](reference/shapes.md#jax-shape-002) |
| Type promotion or dtype | [JAX-TYPE-001](reference/shapes.md#jax-type-001) |

A generic backend exception can route to a more specific guide when the message contains a recognized symptom. Inspect the original cause and device status before deciding what to change.
