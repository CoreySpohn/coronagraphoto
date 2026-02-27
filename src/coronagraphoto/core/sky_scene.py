"""Sky scene object that wraps an exoverses.jax System with background sources."""

from typing import Optional

import equinox as eqx
from exoverses.jax import System

from coronagraphoto.core.zodi_sources import AbstractZodiSource


class SkyScene(eqx.Module):
    """Container for a planetary system and background sources.

    Wraps an ``exoverses.jax.System`` (star + planets + disk) and adds
    observatory-dependent backgrounds like zodiacal light.

    Attributes:
        system: The astrophysical system (star, planets, disk).
        zodi: Optional zodiacal light source.
    """

    system: System

    # Zodiacal light (any ZodiSource variant: ZodiSourceAYO, ZodiSourceLeinert, etc.)
    # Will eventually be replaced by orbix Observatory-based zodi
    zodi: Optional[AbstractZodiSource] = None

    # ── Convenience accessors for backwards compatibility ──

    @property
    def stars(self):
        """Access the star (backwards compat with old SkyScene)."""
        return self.system.star

    @property
    def star(self):
        """Access the star."""
        return self.system.star

    @property
    def planets(self):
        """Access the planets."""
        return self.system.planet

    @property
    def planet(self):
        """Access the planets."""
        return self.system.planet

    @property
    def disk(self):
        """Access the disk."""
        return self.system.disk
