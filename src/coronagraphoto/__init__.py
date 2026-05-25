"""Coronagraphic image simulation built on JAX.

Scene primitives (Star, Planet, Disk, System, Scene, backgrounds) live in
:mod:`skyscapes`. coronagraphoto consumes a :class:`skyscapes.Scene` and
produces either deterministic count-rate maps (``*_rate`` functions,
differentiable, used for fitting and retrievals) or Poisson-realised
detector readouts (``*_readout`` functions, used for data generation).
"""

from optixstuff import OpticalPath

from coronagraphoto.loaders import load_scene_from_exovista
from coronagraphoto.optical_elements import (
    IdealDetector,
    SimplePrimary,
)
from coronagraphoto.simulation import (
    disk_rate,
    disk_readout,
    planet_rate,
    planet_readout,
    star_rate,
    star_readout,
    system_rate,
    system_readout,
    zodi_rate,
    zodi_readout,
)

__all__ = [
    "IdealDetector",
    "OpticalPath",
    "SimplePrimary",
    "disk_rate",
    "disk_readout",
    "load_scene_from_exovista",
    "planet_rate",
    "planet_readout",
    "star_rate",
    "star_readout",
    "system_rate",
    "system_readout",
    "zodi_rate",
    "zodi_readout",
]
