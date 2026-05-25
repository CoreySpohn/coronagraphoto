# Installation

## Basic install

```bash
pip install coronagraphoto
```

This pulls in coronagraphoto and its JAX-CPU dependencies. For most
analysis tasks (sensitivity studies, small-scale simulations) the CPU
build is fine.

## GPU install

GPU acceleration matters when generating mission-scale datasets or
running large vmaps over wavelength / time. JAX with CUDA 12:

```bash
pip install coronagraphoto jax[cuda12]
```

On a fresh Linux machine with a CUDA-capable GPU, that single command
gets you running. On a shared system you may need to point JAX at the
right CUDA runtime; see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for details.

### Google Colab (RTD / Jupyter)

Colab images come with their own (often outdated) JAX. On every Colab
session, force a matched install **before any other JAX-using import**:

```python
%pip install --upgrade --quiet "jax[cuda12]"
```

Then **Runtime > Restart session**. After restart, verify x64 actually
loaded:

```python
import jax
print(jax.default_backend())             # 'gpu'
print(jax.config.read("jax_enable_x64")) # True if you enabled it
```

If `default_backend()` reports `gpu` but the CUDA plugin warned about
version mismatch during install, JAX may have silently fallen back to
a degraded path. The most reliable fix is to restart the kernel after
the `pip install --upgrade jax[cuda12]` step.

## Working from source

For development:

```bash
git clone https://github.com/CoreySpohn/coronagraphoto
cd coronagraphoto
uv sync --all-packages
```

`uv` and `--all-packages` ensure every workspace dependency (yippy,
skyscapes, optixstuff, ...) installs as an editable workspace member.

## Sibling libraries

coronagraphoto depends on several libraries that hold parts of the
forward model:

| Package | Role |
|---|---|
| `skyscapes` | `Scene`, `Star`, `Planet`, `Disk`, physical models |
| `optixstuff` | `OpticalPath`, detectors, throughput, filters |
| `yippy` | YIP-based PSF synthesis (the coronagraph backend) |
| `hwoutils` | Shared unit conversions, transforms, JAX configuration |

`pip install coronagraphoto` pulls all of these as transitive
dependencies. If you `uv sync` from source, use `--all-packages` to
get the workspace-editable versions.

## Verifying the install

```python
import coronagraphoto
print(coronagraphoto.__version__)
```

The first JAX import takes 10-20 s on cold cache as XLA initializes.
This is normal.
