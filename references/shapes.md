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

Under JIT, choose by result shape:

- Fixed window, runtime start: use `lax.dynamic_slice` with static sizes. Check clamping.
- Runtime extent in a fixed buffer: compare broadcasted `jnp.arange` on each axis with the extent. Use `jnp.where` to preserve valid entries and fill the rest, or mask contributions before reducing.
- Variable-length output: return padding and a valid length, or select outside JIT.

Test zero, full, and mixed extents under JIT. Check shape, fill, and gradients.

[JAX dynamic slicing](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html)
