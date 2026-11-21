"""Application-level reproductions adapted from public JAX ecosystem reports."""

CASES = {
    "masked-angle": {
        "source": "https://github.com/jax-ml/jax/issues/1052",
        "docs": "jax.numpy.where",
        "error": "non-finite angle gradient",
        "task": "Preserve angle_loss(y, x, active, weight). It sums weight * atan2(y,x) at active entries, and inactive entries contribute exactly zero value and zero gradient, including at (0,0). Active entries are nonsingular. Preserve JIT, batching, and derivatives with respect to both coordinate arrays and weights.",
        "program": """import jax.numpy as jnp

def angle_loss(y, x, active, weight):
    return jnp.where(active, weight * jnp.arctan2(y, x), 0).sum()
""",
        "smoke": """import jax, jax.numpy as jnp
from solution import angle_loss

y, x = jnp.array([0., 1.]), jnp.array([0., 2.])
a, w = jnp.array([False, True]), jnp.array([1., 2.])
value, (dy, dx) = jax.jit(jax.value_and_grad(angle_loss, argnums=(0, 1)))(y, x, a, w)
assert bool(jnp.all(jnp.isfinite(dy)) & jnp.all(jnp.isfinite(dx))), "non-finite angle gradient"
assert float(dy[0]) == float(dx[0]) == 0.
assert abs(float(value) - 0.9272952) < 1e-6
""",
        "verify": """import jax, jax.numpy as jnp
import numpy as np
from solution import angle_loss

run = jax.jit(jax.vmap(jax.value_and_grad(angle_loss, argnums=(0, 1, 3))))
y = np.array([[0., 1., -2., 8.], [0., 0., 4., 0.], [2., -1., 0., 1.]], dtype=np.float32)
x = np.array([[0., 2., -3., 3.], [0., 0., 2., 0.], [-3., -2., 2., 0.]], dtype=np.float32)
a = np.array([[False, True, True, False], [False]*4, [True]*4])
w = np.array([[1., 2., -3., 4.], [2., 1., 3., 5.], [1., -1., 0., 2.]], dtype=np.float32)
value, (dy, dx, dw) = run(y, x, a, w)
angle = np.where(a, np.arctan2(y, x), 0)
r2 = np.where(a, x*x + y*y, 1)
np.testing.assert_allclose(value, (w * angle).sum(axis=1), atol=1e-6)
np.testing.assert_allclose(dy, np.where(a, w*x/r2, 0), atol=1e-6)
np.testing.assert_allclose(dx, np.where(a, -w*y/r2, 0), atol=1e-6)
np.testing.assert_allclose(dw, angle, atol=1e-6)
""",
        "reference": """import jax.numpy as jnp
def angle_loss(y, x, active, weight):
    angle = jnp.arctan2(jnp.where(active, y, 0), jnp.where(active, x, 1))
    return jnp.where(active, weight * angle, 0).sum()
""",
    },
    "batched-padding": {
        "source": "https://github.com/jax-ml/jax/discussions/22458",
        "docs": "jax.lax.dynamic_slice",
        "error": "Slice entries must be static integers",
        "task": "Preserve fill_padding(values, sizes, fill). values has shape (batch, *padded_shape), sizes contains one runtime valid extent per spatial axis per example, and fill has one scalar per example. Keep entries in each prefix rectangle and replace other entries with the corresponding fill. Support arbitrary spatial rank of at least one, zero extents, JIT, and gradients with respect to values and fill.",
        "program": """import jax
import jax.numpy as jnp

def fill_padding(values, sizes, fill):
    def one(value, size, pad):
        region = tuple(slice(0, n) for n in size)
        valid = jnp.zeros(value.shape, dtype=bool).at[region].set(True)
        return jnp.where(valid, value, pad)
    return jax.vmap(one)(values, sizes, fill)
""",
        "smoke": """import jax, jax.numpy as jnp
from solution import fill_padding
x = jnp.arange(8., dtype=jnp.float32).reshape(2, 4)
actual = jax.jit(fill_padding)(x, jnp.array([[2], [1]]), jnp.array([-1., -2.]))
assert actual.tolist() == [[0., 1., -1., -1.], [4., -2., -2., -2.]]
""",
        "verify": """import jax, jax.numpy as jnp
import numpy as np
from solution import fill_padding

def loss(x, sizes, fill):
    return fill_padding(x, sizes, fill).sum()

for shape, sizes in [((4,), [[0], [4], [2]]),
                     ((3, 5), [[3, 2], [0, 4], [2, 5]]),
                     ((2, 3, 4), [[1, 3, 2], [2, 0, 4], [2, 3, 4]])]:
    values = np.arange(3 * np.prod(shape), dtype=np.float32).reshape((3, *shape))
    fill = np.array([-2., -9., -1.], dtype=np.float32)
    mask = np.zeros(values.shape, dtype=bool)
    for i, size in enumerate(sizes):
        mask[(i, *(slice(0, n) for n in size))] = True
    expected = np.where(mask, values, fill.reshape((-1,) + (1,) * len(shape)))
    actual = jax.jit(fill_padding)(values, jnp.array(sizes), fill)
    dv, df = jax.jit(jax.grad(loss, argnums=(0, 2)))(values, jnp.array(sizes), fill)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(dv, mask)
    np.testing.assert_array_equal(df, (~mask).reshape(3, -1).sum(axis=1))
""",
        "reference": """import jax, jax.numpy as jnp
def fill_padding(values, sizes, fill):
    def one(value, size, pad):
        valid = jnp.ones(value.shape, dtype=bool)
        for axis, extent in enumerate(value.shape):
            shape = [1] * value.ndim
            shape[axis] = extent
            valid &= jnp.arange(extent).reshape(shape) < size[axis]
        return jnp.where(valid, value, pad)
    return jax.vmap(one)(values, sizes, fill)
""",
    },
    "bounded-solver-vjp": {
        "source": "https://github.com/google-deepmind/mujoco/issues/2259",
        "docs": "jax.lax.while_loop",
        "error": "Reverse-mode differentiation does not work",
        "task": "Preserve solve(initial, target, rate, tolerance, limit), returning (state, steps) for the recurrence state += rate * (target-state). Stop before the next update if any stopping condition holds: steps reaches limit, steps reaches hard cap 16, or abs(target-state) <= tolerance. All arguments including limit are dynamic under JIT. Preserve reverse-mode derivatives of state with respect to initial, target, and rate, holding the chosen iteration count fixed locally.",
        "program": """import jax
import jax.numpy as jnp

def solve(initial, target, rate, tolerance, limit):
    def cond(carry):
        state, steps = carry
        return ((steps < limit) & (steps < 16) & (jnp.abs(target-state) > tolerance))
    def body(carry):
        state, steps = carry
        return state + rate * (target-state), steps + 1
    return jax.lax.while_loop(cond, body, (initial, jnp.int32(0)))
""",
        "smoke": """import jax
from solution import solve
assert abs(float(jax.jit(jax.grad(lambda x: solve(x, 2., .25, .01, 3)[0]))(0.)) - .421875) < 1e-6
""",
        "verify": """import jax, jax.numpy as jnp
import numpy as np
from solution import solve

run = jax.jit(jax.vmap(solve))
diff = jax.jit(jax.vmap(jax.grad(lambda *a: solve(*a)[0], argnums=(0, 1, 2))))
cases = [(0., 2., .25, .01, 3), (2., 2., .5, .05, 10), (2., 4., .5, .05, 0),
         (0., 1., .5, .13, 9), (-2., 1., .1, .0002, 99), (2., -1., .3, .7, 10)]
args = tuple(jnp.array(v) for v in zip(*cases))
values, counts = run(*args)
grads = diff(*args)
for i, (initial, target, rate, tolerance, limit) in enumerate(cases):
    state, n = initial, 0
    while n < min(16, limit) and abs(target - state) > tolerance:
        state += rate * (target - state)
        n += 1
    assert int(counts[i]) == n
    np.testing.assert_allclose(values[i], state, atol=1e-6)
    expected = ((1-rate)**n, 1-(1-rate)**n,
                (target-initial)*n*(1-rate)**(n-1) if n else 0.)
    np.testing.assert_allclose([float(g[i]) for g in grads], expected, atol=2e-6)
""",
        "reference": """import jax, jax.numpy as jnp
def solve(initial, target, rate, tolerance, limit):
    def step(_, carry):
        state, count = carry
        active = (count < limit) & (jnp.abs(target-state) > tolerance)
        return jnp.where(active, state + rate*(target-state), state), count + active.astype(jnp.int32)
    return jax.lax.fori_loop(0, 16, step, (initial, jnp.int32(0)))
""",
    },
    "closed-over-implicit-vjp": {
        "source": "https://github.com/google/jaxopt/issues/31",
        "docs": "jax.custom_vjp",
        "error": "UnexpectedTracerError",
        "task": "Repair implicit_solution(scale, data, initial). Preserve the 12 finite primal iterations in solve and their dependence on initial. Preserve the deliberate implicit reverse-mode semantics of the fixed point: for upstream g, VJP is (-g * data / scale**2, g / scale, 0) for (scale, data, initial). A plain autodiff gradient of the 12 primal iterations gives a different answer and is incorrect for this contract. Support batching and JIT.",
        "program": """import jax

def implicit_solution(scale, data, initial):
    @jax.custom_vjp
    def solve(start):
        return jax.lax.fori_loop(0, 12, lambda _, z: z + .15 * (data-scale*z), start)

    def forward(start):
        return solve(start), ()

    def backward(residual, g):
        return (0. * g,)

    solve.defvjp(forward, backward)
    return solve(initial)
""",
        "smoke": """import jax
from solution import implicit_solution
state, (ds, dd, di) = jax.jit(jax.value_and_grad(implicit_solution, argnums=(0, 1, 2)))(2., 6., 1.)
assert abs(float(ds) + 1.5) < 1e-6
assert abs(float(dd) - .5) < 1e-6
assert float(di) == 0. and abs(float(state) - 3.) > 1e-3
""",
        "verify": """import jax, jax.numpy as jnp
import numpy as np
from solution import implicit_solution

scale = jnp.array([1.2, 2., 3.1])
data = jnp.array([4., -3., 1.])
initial = jnp.array([-5., 10., .1])
upstream = jnp.array([2., -.5, 3.])
run = jax.jit(jax.vmap(implicit_solution))
actual, pullback = jax.vjp(run, scale, data, initial)
expected = np.asarray(initial)
for _ in range(12):
    expected = expected + .15 * (np.asarray(data) - np.asarray(scale) * expected)
np.testing.assert_allclose(actual, expected, atol=2e-6)
for grad, reference in zip(pullback(upstream), (-upstream * data / scale**2, upstream / scale, jnp.zeros_like(initial))):
    np.testing.assert_allclose(grad, reference, atol=2e-6)
""",
        "reference": """import jax, jax.numpy as jnp
def primal(scale, data, initial):
    return jax.lax.fori_loop(0, 12, lambda _, z: z + .15*(data-scale*z), initial)
implicit_solution = jax.custom_vjp(primal)
def forward(scale, data, initial):
    return primal(scale, data, initial), (scale, data)
def backward(residual, g):
    scale, data = residual
    return -g*data/scale**2, g/scale, jnp.zeros_like(g)
implicit_solution.defvjp(forward, backward)
""",
    },
}
