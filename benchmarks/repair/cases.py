"""Reduced reproductions of public JAX reports and independent checks."""

CASES = {
    "convergence-loop": {
        "source": "https://github.com/jax-ml/jax/discussions/19650",
        "expected_error": "TracerBoolConversionError",
        "docs": "jax.errors.TracerBoolConversionError",
        "task": (
            "Preserve power_iteration(matrix, initial, max_steps, tolerance). "
            "It must normalize after each matrix-vector product and stop after the first "
            "update whose norm difference from the previous vector is below tolerance, "
            "or after max_steps. For zero steps, return the initial vector. Keep it usable "
            "with JIT and a runtime integer max_steps and float tolerance."
        ),
        "program": """import jax
import jax.numpy as jnp

@jax.jit
def power_iteration(matrix, initial, max_steps, tolerance):
    vector = initial
    for _ in range(32):
        candidate = matrix @ vector
        candidate = candidate / jnp.linalg.norm(candidate)
        delta = jnp.linalg.norm(candidate - vector)
        vector = candidate
        if delta < tolerance:
            break
    return vector
""",
        "repro": """
import jax.numpy as jnp
from solution import power_iteration
power_iteration(jnp.diag(jnp.array([3., 1.])), jnp.array([0.6, 0.8]), 12, 0.01)
""",
        "verify": """
import jax
import jax.numpy as jnp
import numpy as np
from solution import power_iteration

@jax.jit
def run(matrix, initial, limit, tol):
    return power_iteration(matrix, initial, limit, tol)

matrix = np.array([[2.0, 0.4], [0.4, 1.5]], dtype=np.float32)
initial = np.array([0.6, 0.8], dtype=np.float32)
for limit, tol in [(0, 0.0), (1, 0.0), (2, 2.0), (7, 0.0), (12, 0.02)]:
    expected = initial.copy()
    for _ in range(limit):
        candidate = matrix @ expected
        candidate /= np.linalg.norm(candidate)
        delta = np.linalg.norm(candidate - expected)
        expected = candidate
        if delta < tol:
            break
    actual = run(matrix, initial, jnp.int32(limit), jnp.float32(tol))
    np.testing.assert_allclose(actual, expected, atol=1e-6)
""",
    },
    "matrix-chain": {
        "source": "https://github.com/jax-ml/jax/discussions/5576",
        "expected_error": "TracerArrayConversionError",
        "docs": "jax.errors.TracerArrayConversionError",
        "task": (
            "Preserve evolve(matrices, vector), applying each square matrix in order "
            "to the vector. Handle an empty matrix stack and preserve JIT and "
            "differentiability with respect to the vector."
        ),
        "program": """import jax
import numpy as np

@jax.jit
def evolve(matrices, vector):
    for index in range(matrices.shape[0]):
        vector = np.dot(matrices[index], vector)
    return vector
""",
        "repro": """
import jax.numpy as jnp
from solution import evolve
evolve(jnp.arange(8., dtype=jnp.float32).reshape(2, 2, 2), jnp.ones(2))
""",
        "verify": """
import jax
import jax.numpy as jnp
import numpy as np
from solution import evolve

vector = jnp.array([1.5, -2.0])
base = np.array([[[1., 2.], [0., 2.]], [[2., 0.], [1., 1.]], [[0., 1.], [3., 0.]]])
for n in (0, 1, 3):
    matrices = jnp.asarray(base[:n], dtype=jnp.float32)
    product = np.eye(2)
    for matrix in base[:n]:
        product = matrix @ product
    actual = jax.jit(evolve)(matrices, vector)
    derivative = jax.jit(jax.grad(lambda v: evolve(matrices, v).sum()))(vector)
    np.testing.assert_allclose(actual, product @ vector)
    np.testing.assert_allclose(derivative, product.sum(axis=0))
""",
    },
    "dynamic-prefix": {
        "source": "https://github.com/jax-ml/jax/discussions/23087",
        "expected_error": "Slice entries must be static integers",
        "docs": "jax.lax.dynamic_slice",
        "task": (
            "Preserve score_prefixes(weights, states, lengths). weights is a matrix, "
            "states has one scalar per column, and lengths supplies an integer prefix "
            "length per query. Return one row per query with weights[:, :length] @ "
            "states[:length]. Lengths are runtime values in [0, number of columns]. "
            "Keep the function JIT compatible and differentiable with respect to states."
        ),
        "program": """import jax
import jax.numpy as jnp

@jax.jit
def score_prefixes(weights, states, lengths):
    def step(carry, length):
        score = weights[:, :length] @ states[:length]
        return carry, score

    return jax.lax.scan(step, None, lengths)[1]
""",
        "repro": """
import jax.numpy as jnp
from solution import score_prefixes
score_prefixes(jnp.ones((2, 4)), jnp.arange(4.), jnp.array([0, 2, 4]))
""",
        "verify": """
import jax
import jax.numpy as jnp
import numpy as np
from solution import score_prefixes

for width in (3, 5):
    weights = np.arange(2 * width, dtype=np.float32).reshape(2, width) - 3
    vector = jnp.arange(width, dtype=jnp.float32) - 1
    for lengths in ([0, width, 1], [2, 1, width - 1]):
        expected = np.stack([weights[:, :n] @ np.asarray(vector)[:n] for n in lengths])
        actual = jax.jit(score_prefixes)(weights, vector, jnp.array(lengths))
        gradient = jax.jit(jax.grad(lambda v: score_prefixes(weights, v, jnp.array(lengths)).sum()))(vector)
        expected_gradient = np.zeros(width)
        for n in lengths:
            expected_gradient[:n] += weights[:, :n].sum(axis=0)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(gradient, expected_gradient)
""",
    },
    "empty-prng-batch": {
        "source": "https://github.com/jax-ml/jax/issues/37859",
        "expected_error": "KeyReuseError",
        "docs": "jax.errors.KeyReuseError",
        "task": (
            "Preserve sample_batch(keys) and its draw function. For nonempty typed key "
            "batches, produce exactly the same draws as vectorizing draw. For an empty "
            "key batch, return an empty float32 array. Support JIT and leave "
            "jax_debug_key_reuse enabled. Target JAX 0.10.0 without changing dependencies."
        ),
        "program": """import jax
from jax import random

def draw(key):
    a, b = random.split(key)
    return random.normal(a) + random.normal(b)

def sample_batch(keys):
    return jax.vmap(draw)(keys)
""",
        "repro": """
import jax
from jax import random
from solution import sample_batch
jax.config.update("jax_debug_key_reuse", True)
sample_batch(random.split(random.key(0), 0))
""",
        "verify": """
import jax
from jax import random
import numpy as np
from solution import sample_batch

jax.config.update("jax_debug_key_reuse", True)

def reference(key):
    a, b = random.split(key)
    return random.normal(a) + random.normal(b)

run = jax.jit(sample_batch)
reference_run = jax.jit(jax.vmap(reference))
for seed in (5, 19):
    empty = run(random.split(random.key(seed), 0))
    assert empty.shape == (0,) and empty.dtype == np.float32
    for size in (1, 5):
        expected = reference_run(random.split(random.key(seed), size))
        actual = run(random.split(random.key(seed), size))
        np.testing.assert_array_equal(actual, expected)
assert jax.config.jax_debug_key_reuse
""",
    },
}
