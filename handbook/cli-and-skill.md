# CLI and agent skill

## Command line

Install the Python package to get the `jax-diagnose` command. It accepts a string or reads a traceback from standard input.

```sh
jax-diagnose TracerBoolConversionError
jax-diagnose --json < traceback.txt
```

A route returns status 0. An unmatched input returns 1 and produces `{"match": null}` in JSON mode. The checkout command `python3 scripts/diagnose.py` also runs on Python 3.10.

## Agent skill

Install in your project with Node.js 22.20 or newer.

```sh
npx skills add mlnomadpy/jax-diagnostics
```

To select Codex without prompts, add `--agent codex --yes`. In Codex, invoke `$jax-diagnostics` and provide the original error, failing transform or operation, backend, versions, and relevant input shapes. Use the guide's verification step on the failing case. [Skills.sh listing](https://skills.sh/mlnomadpy/jax-diagnostics/jax-diagnostics)

The [skill entry point](https://github.com/mlnomadpy/jax-diagnostics/blob/main/SKILL.md) can also be read directly. Its reference guides work as documentation without running Python.
