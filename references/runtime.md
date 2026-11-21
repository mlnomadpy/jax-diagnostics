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

## JAX-RUNTIME-001

A backend reported an execution or compilation failure. Read the complete backend status and select a narrower route when it identifies OOM, sharding, kernel lowering, or device initialization. Synchronize a minimal failing call with `block_until_ready()` to distinguish where asynchronous execution fails from where the error is surfaced. Record device, versions, operation and backend status before retrying.

[JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)

## JAX-KERNEL-001

Separate lowering errors from runtime faults. For lowering, reduce to one kernel and inspect operand layouts, block shapes and backend restrictions. For runtime faults, compare a small supported interpret-mode case against a pure JAX reference, then verify the actual kernel on its target backend. For DMA or semaphore hangs, check send/receive pairing, ordering, and buffer lifetime. Interpret mode can miss target-specific behavior.

[Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html) · [TPU Pallas debugging skill](https://github.com/alpharomercoma/xla-agentic-development/blob/main/skills/tpu-pallas-debugging/SKILL.md)

## JAX-DIST-001

Collect the earliest failure across ranks before treating the final timeout as the cause. For initialization, check process count, unique IDs, coordinator reachability and initialization before device access. For collectives, check call order, mesh and shapes on all ranks. Verify with a bounded collective on the same topology before rerunning a workload.

[Multi-controller JAX](https://docs.jax.dev/en/latest/multi_process.html)

## JAX-SHARD-001

Inspect actual layouts with `jax.debug.visualize_array_sharding` or `jax.debug.inspect_array_sharding` during lowering. Compare mesh axes, partition specs and global shapes at the failing boundary. Check divisibility and distinguish global sizes from process-local batches. Validate expected per-device shapes and compare against a small unsharded reference.

[JAX distributed arrays](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)

## JAX-TRANSFER-001

Identify which operation caused a transfer across the guarded boundary. Typical triggers include NumPy conversion, scalar extraction and implicit transfers when combining host values with device arrays. Move an intentional transfer to an explicit boundary and rerun with the same transfer guard enabled.

[JAX transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html)

## JAX-BUFFER-001

Check whether a donated argument, a stale alias, or an explicitly deleted array was reused. Thread the returned replacement through the caller. Remove donation when the original value must remain live. Verify repeated transformed calls and any branch that reuses the original value.

[JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)

## JAX-BACKEND-001

Record JAX, jaxlib and plugin versions, backend selection and visible devices. Check the first loader or initialization error for a missing library or incompatible plugin. Compare the environment with JAX's installation matrix. Verify the intended platform using device discovery and a synchronized small operation on that platform.

[JAX installation](https://docs.jax.dev/en/latest/installation.html)
