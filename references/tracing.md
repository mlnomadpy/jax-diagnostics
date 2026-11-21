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

## JAX-TRACE-003

A traced predicate reached Python `if`, `while`, `and`, `or`, `not`, `bool`, or a built-in that compares values. Use `lax.cond` for scalar data-dependent branching, JAX logical operators for array predicates, or `lax.while_loop` for a dynamic loop with fixed carry shape. Verify both branches or loop exit paths under the failing transform. Array masking can evaluate both branches and requires care with undefined gradients.

[JAX boolean conversion](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError)

## JAX-TRACE-004

A Python API tried to turn a tracer into an integer, often for a list index, axis, or loop bound. Use a JAX operation with a dynamic index and static result size, or make a true configuration argument static. Verify a second valid value and check for unintended recompiles when using static arguments.

[JAX integer conversion](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerIntegerConversionError)

## JAX-TRACE-005

A tracer reached NumPy or another host library through `__array__`. Migrate the operation to JAX, move host work outside the transform, or use a callback with an explicit result contract when host execution is required. Convert a host lookup array to a JAX array before indexing it with a tracer. Verify under the original transformation.

[JAX array conversion](https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerArrayConversionError)

## JAX-TRACE-006

A boolean mask selected a runtime-dependent number of values. Preserve shape with masked math or `jnp.where` when possible. For gathering into fixed capacity, propagate validity and define overflow behavior. For masked updates, use a shape-preserving expression. Test all-false, all-true, and mixed masks under JIT.

[JAX boolean indexing](https://docs.jax.dev/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError)
