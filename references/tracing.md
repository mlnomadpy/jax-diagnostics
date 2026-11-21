# Tracing

## JAX-TRACE-001

A transformed value reached a Python operation that needs a concrete value, or an array operation needs a statically known shape.

- For a traced predicate in Python control flow, use `jax.lax.cond` when the predicate is data-dependent. Both branches must return compatible pytrees with matching shapes and dtypes.
- For boolean indexing that produces a variable number of elements, use fixed-shape masking when the downstream operation allows it. If padding to a fixed capacity, carry the valid count or mask and handle overflow explicitly.
- For NumPy array conversion inside the transform, use the equivalent JAX operation or place host processing outside the transform. For a genuine configuration argument that controls shape or Python branching, consider marking it static and check the impact on recompilation.

Verify the original transformed call with multiple input values or both predicate branches. Include empty and full masks where applicable.

[JAX errors](https://docs.jax.dev/en/latest/errors.html)

## JAX-TRACE-002

A transformed function let a tracer escape through a global, closure, object mutation, or container append.

Reproduce with `jax.checking_leaks()` around the smallest failing transformed call. Identify the reference that retains the tracer and return the needed values explicitly or thread the state through the transform. Check the original repeated-call sequence and run the leak checker on it again.

[JAX tracer leaks](https://docs.jax.dev/en/latest/errors.html#jax.errors.UnexpectedTracerError)
