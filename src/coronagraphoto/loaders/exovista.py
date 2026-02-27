"""ExoVista data loader — delegates to exoverses.jax and wraps in SkyScene."""

from collections.abc import Sequence

from exoverses.jax import from_exovista

from coronagraphoto.core.sky_scene import SkyScene
from coronagraphoto.core.zodi_sources import ZodiSourceAYO


def load_sky_scene_from_exovista(
    fits_file: str,
    planet_indices: Sequence[int] | None = None,
    required_planets: int | None = None,
    only_earths: bool = False,
) -> SkyScene:
    """Load complete sky scene from ExoVista FITS file.

    Delegates system loading to ``exoverses.jax.from_exovista()`` and
    wraps the result in a :class:`SkyScene` with a default AYO zodi source.

    Args:
        fits_file: Path to the ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        required_planets: Pad/truncate to this many planets for fixed shapes.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.

    Returns:
        SkyScene object containing the system and zodiacal light source.
    """
    system = from_exovista(
        fits_file,
        planet_indices=planet_indices,
        required_planets=required_planets,
        only_earths=only_earths,
    )

    # Create ZodiSource with AYO-compatible default (22 mag/arcsec² at V)
    zodi = ZodiSourceAYO(
        wavelengths_nm=system.star._wavelengths_nm,
        surface_brightness_mag=22.0,
    )

    return SkyScene(system=system, zodi=zodi)
