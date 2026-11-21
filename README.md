# JAX Diagnostics

A standalone agent skill that routes JAX and accelerator failures to focused repair guides. Inspired by [Solid 2.0 diagnostics](https://www.solidjs.com/blog/solid-2-0-the-long-hold/).

Start with the [diagnostic index](SKILL.md). Give your agent the exception, the failing transform or operation, and the backend. For example, a `TracerBoolConversionError` points to [JAX-TRACE-003](references/tracing.md#jax-trace-003) and a TPU-only NaN points to [JAX-NUM-001](references/numerics.md#jax-num-001).

To use it as a Codex skill, clone this repository into `~/.codex/skills/jax-diagnostics` and invoke `$jax-diagnostics` with your error or symptom. The skill is documentation and works without installing JAX, Flax, or Python. Existing JAX errors and application code can be used as-is.

Each guide gives a trigger, a way to distinguish likely causes, and a check for the repair. The diagnostic IDs belong to this repository.

## Add diagnostics to a training entry point

Install from a checkout of this repository with Python 3.11 or newer.

```sh
python -m pip install .
```

In your training project, wrap the existing entry point outside `jax.jit`.

```python
from jax_diagnostics import annotate_errors

if __name__ == "__main__":
    with annotate_errors():
        main()
```

A matched failure includes the diagnostic ID and guide URL in its traceback. The context manager re-raises the original exception. It only runs diagnostics when an exception exits the block. Pass `guide_base` to use a pinned revision or a mirror that serves the guide files. The default URLs use this GitHub repo and require access while the repo is private.

## Route an error from the terminal

```sh
python3 scripts/diagnose.py TracerBoolConversionError
python3 scripts/diagnose.py --json < traceback.txt
```

The checkout CLI uses Python 3.10 or newer and the standard library. An installed package also exposes `jax-diagnose`. It returns exit status 1 for an unrecognized failure. Symptom matches are starting points for diagnosis.

The catalog covers all 12 named exceptions in the [pinned public JAX error module](https://github.com/jax-ml/jax/blob/7261b7bfe9c6e75d04a2e36f0387977def874fe0/jax/errors.py), plus common failures that use generic exceptions or backend status text. JAX operations, third-party libraries and device runtimes can also emit other messages.

Run `python3 -m unittest discover -s tests` to verify routing and guide links.
