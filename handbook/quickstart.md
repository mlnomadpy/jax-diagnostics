# Quickstart

Use Python 3.11 or newer. Your training project supplies its own JAX installation and accelerator runtime.

Install from [PyPI](https://pypi.org/project/jax-diagnostics/).

```sh
python -m pip install jax-diagnostics
```

Wrap the host-side entry point in `train.py`. This runnable example intentionally puts a traced predicate in Python control flow.

```python
import jax

from jax_diagnostics import annotate_errors


@jax.jit
def sign(flag):
    return 1 if flag else -1


def main():
    print(sign(True))


if __name__ == "__main__":
    with annotate_errors():
        main()
```

Run `python train.py`. The original `TracerBoolConversionError` is followed by `JAX-TRACE-003` and its [repair guide](reference/tracing.md#jax-trace-003). To verify the repair, replace the branch with `jax.lax.cond(flag, lambda: 1, lambda: -1)` and run with both boolean values.

You can also [install the agent skill](cli-and-skill.md#agent-skill) for debugging existing logs.
