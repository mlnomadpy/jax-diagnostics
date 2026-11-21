# Transformations and state

## JAX-TRANSFORM-001

A Jaxpr type check rejected an intermediate program. Reduce to the smallest transform composition and inspect its abstract inputs and outputs. If custom primitives or rules are involved, check that abstract evaluation, lowering, batching, and differentiation agree on shapes, dtypes and effects. Compare behavior with the rule removed or replaced by built-in operations. Verify under the failing transform composition and retain the JAX and jaxlib versions.

[Jaxpr language](https://docs.jax.dev/en/latest/jaxpr.html)

## JAX-RNG-001

One random key reached multiple consuming operations. Trace keys as explicit inputs and outputs. Split for independent uses or use a stable `fold_in` identifier when deriving per-step or per-example keys. Use typed keys and rerun with `jax.debug_key_reuse(True)`. Check both reproducibility for the same seed and distinct keys for independent draws.

[JAX key reuse](https://docs.jax.dev/en/latest/jax.experimental.key_reuse.html)

## JAX-GRAD-001

Check that the differentiated objective returns a scalar and that the selected arguments have the intended differentiable types. Keep auxiliary outputs outside the loss with `has_aux`. For custom derivative rules, check residuals and tangent or cotangent structure. Confirm with a small directional derivative check at a smooth point and replay the original transformed call.

[JAX custom derivative rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)

## JAX-AXIS-001

For `vmap`, compare mapped axis lengths and confirm that `in_axes` matches the input pytree, including intentionally unmapped leaves. For `scan`, inspect the carry tree, shapes and dtypes before and after one iteration. Verify with different batch and sequence lengths to catch an accidental axis swap.

[JAX transformations](https://docs.jax.dev/en/latest/transformations.html)

## JAX-STATE-001

For NNX, use a graph-aware transform when passing or mutating modules, or split graph and array state explicitly at a pure JAX boundary. For Linen, check variable collection scope, mutability, initialization shapes, and RNG stream names. Verify two consecutive calls and compare the expected state updates and RNG behavior.

[Flax NNX transforms](https://flax.readthedocs.io/en/latest/guides/nnx_basics.html)
