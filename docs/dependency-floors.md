# Dependency floors

The lower bounds in `pyproject.toml` are checked with Python 3.12 and
`uv pip install --resolution lowest-direct -e ".[test,docs,dev]"`, applying
`.github/sibling-pins.txt` as `--override`. Build dependencies are also checked
at their exact lower bounds using `--build-constraint`.

Sibling overrides exercise the pinned source commits rather than the released
lower bounds. In particular, the orbix and skyscapes pins retain development
versions below their declared API floors; keep those floors for normal installs.
The canary uses the same overrides as CI.

| Group | Dependency | Lower bound |
| --- | --- | --- |
| Build | hatchling | 1.26.3 |
| Build | hatch-fancy-pypi-readme | 24.1.0 |
| Build | hatch-vcs | 0.4.0 |
| Runtime | hwoutils | 1.10.0 |
| Runtime | jax | 0.8.1 |
| Runtime | optixstuff | 3.0.0 |
| Runtime | orbix | 3.0.0 |
| Runtime | pooch | 1.8.2 |
| Runtime | yippy | 2.10.6 |
| Runtime | skyscapes | 2.1.0 |
| Dev | pre-commit | 3.5.0 |
| Docs | sphinx | 7.2.6 |
| Docs | myst-nb | 1.0.0 |
| Docs | sphinx-book-theme | 1.1.0 |
| Docs | sphinx-autoapi | 3.0.0 |
| Docs | sphinxcontrib-mermaid | 0.9.2 |
| Docs | ipython | 8.18.1 |
| Test | pytest | 7.3.2 |
| Test | hypothesis | 6.79.1 |
| Test | pytest-cov | 1.0 |
| Test | pytest-xdist | 2.5 |
| Test | pytest-testmon | 2.0.7 |
| Test | pytest-markdown-docs | 0.5.0 |
| Test | pytest-timeout | 1.3.1 |
| Test | pytest-fail-slow | 0.1.0 |
| Test | equinox | 0.13.2 |
| Test | lod_unit | 1.8.0 |
| Test | numpy | 2.1.2 |
| Test | packaging | 23.1 |

The NumPy and pytest-markdown-docs bounds match the lowest versions selected
by the all-extras resolution (2.1.2 and 0.5.0). The other non-sibling direct
dependencies resolve exactly to the bounds above. Tests run one file per
pytest process with warnings treated as errors; the scheduled canary selects
the `not slow` tier.

## Open upstream failure

Warnings as errors expose a mixed-precision spline bug when loading the public
`solar_system_mod.fits` demo scene with `only_earths=True`, `load_disk=True`
and JAX x64 enabled. The disk has float64 wavelengths and a float32 contrast
cube. `interpax.CubicSpline` scatters float64 intermediate values into a float32
array and JAX raises its unsafe-cast FutureWarning.

This fails all three tests in `tests/integration/test_exovista_forward.py` and
the test in `tests/integration/test_real_yip_forward.py` in the shared
environment. The warning remains an error and the assertions remain intact.
Both files are in the slow tier; the canary exercises the quick tier.
