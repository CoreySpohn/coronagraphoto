"""Coronagraphic image simulation built on JAX.

Scene primitives (Star, Planet, Disk, System, Scene, backgrounds) live in
:mod:`skyscapes`. coronagraphoto consumes a :class:`skyscapes.Scene` and
turns it into detector images via the ``sim_*`` and ``gen_*`` functions.
"""

from coronagraphoto.core import (
    Exposure,
    OpticalPath,
    sim_disk,
    sim_planets,
    sim_star,
    sim_system,
    sim_zodi,
)
from coronagraphoto.loaders import load_scene_from_exovista
from coronagraphoto.optical_elements import (
    PrimaryAperture,
    SimpleDetector,
)

__all__ = [
    "Exposure",
    "OpticalPath",
    "PrimaryAperture",
    "SimpleDetector",
    "load_scene_from_exovista",
    "sim_disk",
    "sim_planets",
    "sim_star",
    "sim_system",
    "sim_zodi",
]
