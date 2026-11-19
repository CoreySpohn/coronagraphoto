# Dependency floors

The lower bounds in `pyproject.toml` are checked with Python 3.12 and
`uv pip install --resolution lowest-direct -e ".[test,docs,dev]"`, applying
`.github/sibling-pins.txt` as `--override`. Build dependencies are also checked
at their exact lower bounds using `--build-constraint`.

`.github/sibling-pins.txt` lists sibling libraries that are installed from a pinned source
commit while their release is pending; it is empty when PyPI covers every declared floor, and
the canary uses the same overrides as CI.

| Group | Dependency | Lower bound |
| --- | --- | --- |
| Build | hatchling | 1.26.3 |
| Build | hatch-fancy-pypi-readme | 24.1.0 |
| Build | hatch-vcs | 0.4.0 |
| Runtime | equinox | 0.13.2 |
| Runtime | hwoutils | 2.0.0 |
| Runtime | jax | 0.8.1 |
| Runtime | numpy | 2.1.2 |
| Runtime | optixstuff | 6.0.0 |
| Runtime | orbix | 5.0.0 |
| Runtime | yippy | 3.0.0 |
| Runtime | skyscapes | 5.0.0 |
| Runtime | zodi[jax] | 2.0.0 |
| Dev | pre-commit | 3.5.0 |
| Docs | sphinx | 7.2.6 |
| Docs | myst-nb | 1.0.0 |
| Docs | sphinx-book-theme | 1.1.0 |
| Docs | sphinx-autoapi | 3.0.0 |
| Docs | sphinxcontrib-mermaid | 0.9.2 |
| Docs | ipython | 8.18.1 |
| Test | pytest | 7.3.2 |
| Test | hypothesis | 6.79.1 |
| Test | pytest-xdist | 2.5 |
| Test | pytest-testmon | 2.0.7 |
| Test | pytest-markdown-docs | 0.5.0 |
| Test | pytest-timeout | 1.3.1 |
| Test | pytest-fail-slow | 0.1.0 |
| Test | lod_unit | 1.8.0 |
| Test | packaging | 23.1 |

The NumPy and pytest-markdown-docs bounds match the lowest versions selected
by the all-extras resolution (2.1.2 and 0.5.0). The other non-sibling direct
dependencies resolve exactly to the bounds above. Tests run one file per
pytest process with warnings treated as errors; the scheduled canary selects
the `not slow` tier.
