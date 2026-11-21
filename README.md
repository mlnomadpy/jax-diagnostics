# JAX Diagnostics

**Put a repair guide in the traceback.** JAX Diagnostics is a standalone Python library, CLI, and agent skill for JAX errors. It follows the diagnostic pattern in [Solid 2.0](https://www.solidjs.com/blog/solid-2-0-the-long-hold/). A matched failure gets a stable diagnostic ID and a link to the relevant guide.

[Landing page](https://www.tahabouhsine.com/jax-diagnostics/) · [Documentation](https://www.tahabouhsine.com/jax-diagnostics/docs/) · [Blog](https://www.tahabouhsine.com/jax-diagnostics/docs/blog/) · [Agent skill](https://github.com/mlnomadpy/jax-diagnostics/blob/main/SKILL.md) · [Diagnostic guides](#diagnostic-guides)

[![Skills.sh installs](https://skills.sh/b/mlnomadpy/jax-diagnostics)](https://skills.sh/mlnomadpy/jax-diagnostics/jax-diagnostics)

## Install the agent skill

Run this from your project with Node.js 22.20 or newer and select your coding agent when prompted.

```sh
npx skills add mlnomadpy/jax-diagnostics
```

For a non-interactive Codex install, add `--agent codex --yes`. Add `--global` to install across projects. Then invoke `$jax-diagnostics` in Codex with the error and the failing operation or backend. The skill installer installs the guide and its files. The Python wrapper below is installed separately. [Supported agents and installer options](https://github.com/vercel-labs/skills#install-a-skill)

## Use it in training

Use Python 3.11 or newer and install from [PyPI](https://pypi.org/project/jax-diagnostics/).

```sh
python -m pip install jax-diagnostics
```

Wrap the host-side entry point in `train.py` and run it as usual.

```python
from jax_diagnostics import annotate_errors

if __name__ == "__main__":
    with annotate_errors():
        main()
```

For example, a `TracerBoolConversionError` gets this note below the original exception.

```text
[JAX-TRACE-003] Traced Python boolean conversion
Agent repair guide: https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/tracing.md#jax-trace-003
```

The wrapper preserves the exception type, traceback, and exit behavior. It runs diagnostics only when an exception leaves the block. Pass `guide_base` to link to a pinned revision or a mirror. The package has no runtime dependencies.

## Diagnose a saved traceback

```sh
jax-diagnose TracerBoolConversionError
jax-diagnose --json < traceback.txt
```

From a checkout, `python3 scripts/diagnose.py` also works with Python 3.10 or newer. The CLI exits with status 1 when it finds no match.

## Diagnostic guides

All 12 named exceptions in the pinned public `jax.errors` module have dedicated routes. Symptom matches suggest where to start inspecting a failure. [Upstream error inventory](https://github.com/jax-ml/jax/blob/7261b7bfe9c6e75d04a2e36f0387977def874fe0/jax/errors.py)

| Guide | What it covers |
| --- | --- |
| [Tracing](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/tracing.md) | Concretization, tracer escapes, Python and NumPy conversions, boolean indexing |
| [Shapes and types](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/shapes.md) | Symbolic dimensions, broadcasting, dtypes, indexing |
| [Transforms and state](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/transforms.md) | Jaxpr, PRNG key reuse, gradients, mapped axes, Flax state |
| [Numerics and parity](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/numerics.md) | NaNs, infinities, and cross-framework mismatches |
| [Runtime and restore](https://github.com/mlnomadpy/jax-diagnostics/blob/main/references/runtime.md) | Compilation, memory, sharding, collectives, Pallas, and Orbax |

## Development

See the [development and release guide](https://www.tahabouhsine.com/jax-diagnostics/docs/development/) for building the docs, testing the skill, and preparing PyPI artifacts.

## License

[MIT](https://github.com/mlnomadpy/jax-diagnostics/blob/main/LICENSE).
