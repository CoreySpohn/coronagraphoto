<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/coreyspohn/coronagraphoto/main/docs/_static/tmp_logo.png" alt="coronagraphoto logo" />
  <br><br>
</p>

<p align="center">
  <a href="https://pypi.org/project/coronagraphoto/"><img src="https://img.shields.io/pypi/v/coronagraphoto.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://coronagraphoto.readthedocs.io"><img src="https://readthedocs.org/projects/coronagraphoto/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/coronagraphoto/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/coronagraphoto/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/coronagraphoto/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/coronagraphoto/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
  <a href="https://doi.org/10.5281/zenodo.10950736"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10950736-blue?style=flat-square" alt="DOI"/></a>
</p>

---

# coronagraphoto

**coronagraphoto** is a Python library designed to simulate coronagraphic observations of exoplanetary systems. The base "thing" it produces are images/photos, hence the name. It has been designed to bridge the gap between yield calculations and concrete image generation for missions like the Habitable Worlds Observatory (HWO).

The library integrates high-fidelity coronagraph models from the standard format used for yield calculations (dubbed a "Yield Input Package" and loaded via **[yippy](https://github.com/CoreySpohn/yippy)**) with detailed planetary system simulations (via **[ExoVista](https://github.com/alexrhowe/ExoVista)**) to produce realistic detector images.

Built on **JAX**, `coronagraphoto` supports JIT compilation and differentiation of scene predictions with fixed instrument geometry. JAX provides CPU and accelerator execution.

## Key Features

*   **End-to-End Simulation**: From astrophysical scenes to detector readouts.
*   **JAX & JIT Compatible**: High-performance simulations using functional programming patterns.
*   **Modular Design**: flexible optical paths, easily swappable coronagraphs and detectors.
*   **HWO Ready**: Specifically designed to support yield modeling for future direct imaging missions.

## Installation

```bash
pip install coronagraphoto
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements (CUDA/TPU/CPU).)*

## Photon budgets and detector images

Scenes come from `skyscapes`; hardware and detector statistics come from
`optixstuff`. The image reader provides three entry points:

| Function | Output | Units |
|---|---|---|
| `image_rates` | `PhotonRates`, each field `(W, Q, ny, nx)` | photons/s/nm/pixel |
| `image_electrons` | `ElectronRates`, each field `(W, ny, nx)` | electrons/s/pixel |
| `observe` | `Image`, data `(W, ny, nx)` | electrons/pixel |

The fields are `planet`, `star_leakage`, `local_zodi`, `exozodi`, `thermal`
and `stellar_companions`; the reader does not model the last two, so they hold
zeros. Call `.total()`
to combine terms. Optical throughput excludes detector yield. Only
`detector.electrons` applies QE and survival, at the quadrature nodes.
`observe` calls `detector.draw` once on the combined electron budget.

`image_electrons` scans nodes and accumulates band images, holding one node's
images at a time. `image_rates` retains all nodes for inspection and small
problems. Build one quadrature on the host from the active scene and optical
path. Its Gauss-Legendre rule splits at every declared break, including
detector QE knots. Reuse that quadrature across readers.

## Quick start

Enable float64 before loading orbital state vectors. This example downloads
a public scene and YIP on first use; the epoch must lie in the scene's recorded
time range.

```python notest
import jax

jax.config.update("jax_enable_x64", True)

from hwoutils.grids import SkyOrientation
from hwoutils.spectral import Band, Quadrature
from optixstuff import FramePlan, OpticalPath
from skyscapes.datasets import fetch_scene
from yippy import fetch_yip, open_yip
from coronagraphoto import image_electrons, load_scene_from_exovista, observe

scene = load_scene_from_exovista(fetch_scene(), load_disk=False)
coronagraph = open_yip(fetch_yip("eac1_aavc_2d")).prepare()
optical_path = OpticalPath.from_default_setup(
    coronagraph,
    attenuating_throughput=0.4,
    qe=0.8,
    detector_shape=(128, 128),
    pixel_scale_arcsec=0.01,
)
bands = (Band(500.0, 600.0),)
quadrature = Quadrature.build(bands, scene, optical_path)
pointing = SkyOrientation(pa_y_deg=30.0, parity=-1)
options = dict(time_d=0.0, pointing=pointing, zodi=scene.zodi)
rates = image_electrons(scene.system, optical_path, quadrature, **options)
image = observe(
    jax.random.PRNGKey(0),
    scene.system,
    optical_path,
    quadrature,
    FramePlan(frame_time_s=100.0, n_frames=36),
    **options,
)
# imshow(image.data[0], origin="lower")
```

For disks, load the disk and give the coronagraph a scene renderer with
`coronagraph.with_scene_operator()`. The reader requests disk photon radiance on the detector sky grid and passes
its per-pixel flux through the coronagraph's field-dependent PSFs.

Rows increase along detector +y and columns along detector +x. Pixel `(j,i)`
is centred at `(x,y)=(i,j)`, including even grids. `pa_y_deg` is the position
angle of detector +y, north through east. At zero, east is left and north is
up; at 90 degrees, north is right. A roll changes this orientation for both
planets and disks.

Speckle fields may be supplied through `OpticalPath.speckle` or the reader's
`speckle` keyword. Their positive excess joins `star_leakage`; their existing
physics is retained. The reader calls `realize(wavelength_nm=..., time_d=...)`
and leaves epoch conversion to the field. The speckle pattern remains fixed
in detector coordinates as pointing rolls.

Select record fields for per-source access. See
[rate and observation contracts](docs/explanation/rate_vs_readout.md) and
[JAX and memory](docs/explanation/performance.md) for details.
