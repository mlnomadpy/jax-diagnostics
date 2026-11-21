# Development and releases

## Develop locally

```sh
python3 -m unittest discover -s tests
python3 -m pip install "mkdocs==1.6.1"
python3 scripts/build_site.py
python3 -m http.server --directory site
```

The landing page is served at the root and MkDocs at `/docs/`. The build renders the shared reference Markdown from `references/` and includes the handbook URLs in the project sitemap. Add new routes in `jax_diagnostics/_routing.py`, add a stable anchor to the corresponding guide, and add a representative routing fixture.

## Prepare a PyPI release

Work from a clean checkout. Confirm the release version in `pyproject.toml`, the package metadata, and the MIT license metadata before uploading. Check that you can claim the project name on PyPI.

```sh
python -m pip install build twine
python -m unittest discover -s tests
python -m build
python -m twine check --strict dist/*
```

Install the wheel in a fresh environment and check `jax-diagnose TracerBoolConversionError` and the [Python entry point](python-api.md). For a release rehearsal, upload the checked artifacts to TestPyPI and install from there. Then upload those same artifacts to PyPI using the release owner's credentials. Use a fresh output directory for each build to avoid uploading artifacts from earlier versions. [Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

### Publish from GitHub Actions

For the first release, configure a [pending Trusted Publisher](https://pypi.org/manage/account/publishing/) on PyPI with project `jax-diagnostics`, owner `mlnomadpy`, repository `jax-diagnostics`, workflow `pypi.yml`, and environment `pypi`.

Run the [release workflow](https://github.com/mlnomadpy/jax-diagnostics/actions/workflows/pypi.yml) from `main` with the desired version. The default run builds and checks the artifacts. Select **Upload to PyPI using Trusted Publishing** when ready to publish. After release, verify an install from the production index.

```sh
python -m pip install jax-diagnostics==0.1.0
jax-diagnose TracerBoolConversionError
```

## Share on Skills.sh

The public GitHub repository exposes `SKILL.md` at its root with its supporting scripts and references. Check discovery and test an install from a clean project.

```sh
npx skills add mlnomadpy/jax-diagnostics --list
npx skills add mlnomadpy/jax-diagnostics --agent codex --yes
```

The directory discovers public skills through aggregate installer telemetry. Use the [existing listing](https://skills.sh/mlnomadpy/jax-diagnostics/jax-diagnostics) when sharing the skill. [Skills.sh listing process](https://skills.sh/docs/faq)
