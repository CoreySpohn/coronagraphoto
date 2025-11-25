"""Sky scene object that holds collections of sources without control flow."""

import equinox as eqx

from coronagraphoto.core.sources import (
    DiskSource,
    PlanetSources,
    StarSource,
    ZodiSource,
)


class SkyScene(eqx.Module):
    """Container for all sources in a sky scene.

    This class organizes sources by their processing type rather than
    astrophysical type to avoid control flow during simulation.
    """

    # On-axis sources (stars)
    stars: StarSource

    # Off-axis point sources (planets, galaxies)
    planets: PlanetSources

    # Extended sources (disks, exozodiacal light, etc.)
    disk: DiskSource

    # Zodiacal light
    zodi: ZodiSource

    def __init__(
        self,
        stars: StarSource | None = None,
        planets: PlanetSources | None = None,
        disk: DiskSource | None = None,
        zodi: ZodiSource | None = None,
    ):
        """Initialize sky scene with lists of sources.

        All sources expect scalar wavelength and time inputs.
        Use jax.vmap for vectorized evaluation.
        """
        self.stars = stars
        self.planets = planets
        self.disk = disk
        self.zodi = zodi
