# Numerical debugging

## JAX-NUM-001

Keep the failing input, transform, backend, and precision. Find the first NaN or Inf in the forward pass, gradient, or update with `jax_debug_nans`, `checkify`, or targeted checks. Inspect operands for invalid domains, singular gradients, masked reductions, or overflow. Verify the failing input and a nearby valid one under the original settings.

[JAX debugging flags](https://docs.jax.dev/en/latest/debugging/flags.html)

## JAX-PARITY-001

Compare exact input tensors, masks, position IDs, parameter layouts and effective dtypes before comparing outputs. Disable stochastic behavior. Compare token states layer by layer, then pooling and normalization, and trace backwards from the first mismatch. Check absolute error as well as any task-specific metric such as cosine similarity or retrieval rankings.

Keep the tolerance and input cases fixed during diagnosis. Confirm the repaired path on the original failing case and the full declared parity scope before claiming equivalence.

[JAX–PyTorch porting skill](https://github.com/hcsolakoglu/jax-pytorch-porting-skill)
