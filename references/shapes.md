# Shapes, types, and indexing

## JAX-SHAPE-001

Shape polymorphism could not prove a symbolic comparison or constraint. Identify the compared dimension expressions and determine whether the relation holds across the whole declared shape family. Rewrite the expression, specify a valid symbolic constraint, or use a narrower export signature. Verify with multiple admissible concrete sizes and an input outside the contract.

[JAX shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html)

## JAX-SHAPE-002

Record the shape and meaning of each axis at the failing operation with `jax.eval_shape` or a small reproducer. Check batch alignment, trailing-axis broadcasting, contraction axes and reshape element counts. Verify with unequal axis sizes to expose accidental transposes that square inputs conceal.

[JAX rank promotion](https://docs.jax.dev/en/latest/rank_promotion_warning.html)

## JAX-TYPE-001

Find the operation rejecting the type and inspect the operand dtypes, Python scalar weak typing, the X64 setting and the expected signatures. Convert at the intended boundary with an explicit dtype and verify the original computation under the same precision policy. For a specific tracer conversion, use the corresponding tracing route.

[JAX type promotion](https://docs.jax.dev/en/latest/type_promotion.html)

## JAX-INDEX-001

Inspect operand rank, index dtype and shape, traced slice bounds, and required out-of-bounds semantics. Choose an indexing operation that supports the actual index type and has a statically known output shape. Verify boundary indices explicitly. Some JAX gathers and updates handle out-of-bounds indices without raising.

[JAX indexing sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing)

## JAX-INDEX-002

Under JIT, ordinary slice syntax expects static bounds. First identify which index depends on runtime values and whether the result needs a fixed size or a runtime-dependent prefix, rectangle, or window.

- For a fixed-size window at a runtime position, use `lax.dynamic_slice` with static sizes. Check the boundary behavior explicitly. JAX can shift a start index inward to fit the requested window.
- For runtime extents within a fixed buffer, build a mask by comparing each axis's broadcasted `jnp.arange` with its extent. Use `jnp.where` to keep valid entries and fill the rest. For a reduction over a prefix, mask contributions before reducing.
- When a caller truly needs a variable-length array, decide whether to return a padded array with a valid length or move the final variable-length selection outside JIT.

Verify zero, full, and mixed extents in the same compiled function. Check shape, fill values, and gradients when the operation participates in autodiff.

[JAX dynamic slicing](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html)
