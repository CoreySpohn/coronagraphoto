"""ExoVista data loader -- builds a ``skyscapes.Scene`` with a default zodi."""

from collections.abc import Sequence

import jax.numpy as jnp
from skyscapes import Scene, from_exovista
from skyscapes.background import ZodiSourceAYO


def load_scene_from_exovista(
    fits_file: str,
    planet_indices: Sequence[int] | None = None,
    required_planets: int | None = None,
    only_earths: bool = False,
    zodi_surface_brightness_mag: float = 22.0,
) -> Scene:
    """Load a full :class:`skyscapes.Scene` from an ExoVista FITS file.

    Delegates system loading to :func:`skyscapes.from_exovista` and adds
    a default :class:`~skyscapes.background.ZodiSourceAYO` background using
    the host star's wavelength grid.

    Args:
        fits_file: Path to the ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        required_planets: Pad/truncate to this many planets for fixed
            shapes.
        only_earths: If True and ``planet_indices`` is None, auto-filter
            Earths.
        zodi_surface_brightness_mag: V-band surface brightness for the
            default zodi background. Default 22.0 (AYO standard).

    Returns:
        ``skyscapes.Scene`` with the loaded system and a default zodi
        background.
    """
    system = from_exovista(
        fits_file,
        planet_indices=planet_indices,
        required_planets=required_planets,
        only_earths=only_earths,
    )

    wavelengths_nm = jnp.asarray(system.star._wavelengths_nm)
    zodi = ZodiSourceAYO(
        wavelengths_nm=wavelengths_nm,
        surface_brightness_mag=zodi_surface_brightness_mag,
    )

    return Scene(system=system, backgrounds=(zodi,))
