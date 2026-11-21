# Numerical debugging

## JAX-NUM-001

Use the original failing input and preserve backend, precision, masking and compilation settings. Locate the first non-finite tensor in the forward pass, loss, gradient, or optimizer update. For a small JIT reproduction, try `jax_debug_nans`. Use `checkify` or targeted finite assertions to inspect intermediates when the failure only occurs in compiled execution or involves Inf.

Inspect the operands immediately before the first failing operation. Common checks include fully masked reductions, zero denominators, invalid logarithm inputs, and low-precision overflow. A finite eager result can help isolate the compiled path. Verify the repair under the original transform and backend with the failing input and a nearby valid case.

[JAX debugging flags](https://docs.jax.dev/en/latest/debugging/flags.html)

## JAX-PARITY-001

Compare exact input tensors, masks, position IDs, parameter layouts and effective dtypes before comparing outputs. Disable stochastic behavior. Compare token states layer by layer, then pooling and normalization, and trace backwards from the first mismatch. Check absolute error as well as any task-specific metric such as cosine similarity or retrieval rankings.

Keep the tolerance and input cases fixed during diagnosis. Confirm the repaired path on the original failing case and the full declared parity scope before claiming equivalence.

[JAX–PyTorch porting skill](https://github.com/hcsolakoglu/jax-pytorch-porting-skill)
