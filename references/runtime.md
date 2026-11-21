# Compilation, memory and restore

## JAX-COMPILE-001

Enable `jax_log_compiles` and `jax_explain_cache_misses` and rerun a short sequence. Separate tracing or compilation from steady-state execution by timing warmed calls and synchronizing with `block_until_ready()`. Check for changing shapes, dtypes, static arguments, or freshly created jitted functions. After a fix, verify compile counts and warmed timing on the same sequence.

[JAX compile debugging](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)

## JAX-MEM-001

Identify the failing allocation and memory space first. Record per-device shapes and shardings, and distinguish a single large temporary from replication, retained buffers, or the allocator reserving device memory. Inspect the compiled allocation estimate or take a device memory profile when supported. TPU Pallas VMEM errors require checking tile and scratch sizes separately from HBM usage.

Rerun the triggering shape and sharding after the change and compare peak device memory and outputs.

[JAX device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html) · [TPU Pallas debugging skill](https://github.com/alpharomercoma/xla-agentic-development/blob/main/skills/tpu-pallas-debugging/SKILL.md)

## JAX-RESTORE-001

Pin the committed step and inspect checkpoint metadata before restoring. Compare tree paths, shapes, dtypes and requested target sharding. Distinguish a missing checkpoint item from a topology mismatch. For exact training continuation, verify the optimizer state, counters, RNG and input cursor as well as the weights. If testing continuity, compare a short uninterrupted run with a save-and-restore run at the same next step. Keep the source checkpoint available throughout diagnosis.

[Orbax documentation](https://orbax.readthedocs.io/en/latest/)
