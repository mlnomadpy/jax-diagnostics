# JAX Diagnostics

**Put a repair guide in the traceback.** JAX Diagnostics is a standalone Python library, CLI, and agent skill for JAX errors. It follows the diagnostic pattern in [Solid 2.0](https://www.solidjs.com/blog/solid-2-0-the-long-hold/). A matched failure gets a stable diagnostic ID and a link to the relevant guide.

[Landing page](https://www.tahabouhsine.com/jax-diagnostics/) · [Agent skill](SKILL.md) · [Diagnostic guides](#diagnostic-guides)

## Install the agent skill

Run this from your project with Node.js 22.20 or newer and select your coding agent when prompted.

```sh
npx skills add mlnomadpy/jax-diagnostics
```

For a non-interactive Codex install, add `--agent codex --yes`. Add `--global` to install across projects. Then invoke `$jax-diagnostics` in Codex with the error and the failing operation or backend. The skill installer installs the guide and its files. The Python wrapper below is installed separately. [Supported agents and installer options](https://github.com/vercel-labs/skills#install-a-skill)

## Use it in training

Install from GitHub with Python 3.11 or newer.

```sh
python -m pip install "git+https://github.com/mlnomadpy/jax-diagnostics.git"
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
| [Tracing](references/tracing.md) | Concretization, tracer escapes, Python and NumPy conversions, boolean indexing |
| [Shapes and types](references/shapes.md) | Symbolic dimensions, broadcasting, dtypes, indexing |
| [Transforms and state](references/transforms.md) | Jaxpr, PRNG key reuse, gradients, mapped axes, Flax state |
| [Numerics and parity](references/numerics.md) | NaNs, infinities, and cross-framework mismatches |
| [Runtime and restore](references/runtime.md) | Compilation, memory, sharding, collectives, Pallas, and Orbax |

## Development

Run `python3 -m unittest discover -s tests` to verify the router, guides, and exception notes. The landing page is plain HTML, CSS, and JavaScript. Open [docs/index.html](docs/index.html) directly or serve it with `python3 -m http.server --directory docs`.
