"""ExoVista data loader -- builds a ``skyscapes.Scene`` with a default zodi."""

from collections.abc import Sequence

from skyscapes import Scene, from_exovista
from zodi.jax import AYOZodi


def load_scene_from_exovista(
    fits_file: str,
    planet_indices: Sequence[int] | None = None,
    only_earths: bool = False,
    zodi_surface_brightness_mag: float = 22.0,
    load_disk: bool = True,
) -> Scene:
    """Load a full :class:`skyscapes.Scene` from an ExoVista FITS file.

    Delegates system loading to :func:`skyscapes.from_exovista` and adds
    a default :class:`~zodi.jax.AYOZodi` photon background.
    The system retains astronomical ``midplane_position_angle_deg`` and
    stored disk cubes on their native sky grid; no extra disk rotation is
    performed here.

    Enable ``jax.config.update("jax_enable_x64", True)`` before loading:
    imported orbital state vectors need float64 precision. Recent ExoVista
    scenes retain time-dependent planet contrasts and spot metadata; the
    latter does not yet add resolved stellar-surface propagation.

    Args:
        fits_file: Path to the ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        only_earths: If True and ``planet_indices`` is None, auto-filter
            Earths.
        zodi_surface_brightness_mag: AB surface brightness of the default
            zodi background [mag/arcsec^2] at its 550 nm reference
            wavelength. Default 22.0 (AYO standard).
        load_disk: If False, skip the disk extension and set
            ``System.disk`` to ``None``. Use this when the coronagraph
            has no scene renderer: a disk needs one, from
            ``coronagraph.with_scene_operator()``.

    Returns:
        ``skyscapes.Scene`` with the loaded system and a default zodi
        background.
    """
    system = from_exovista(
        fits_file,
        planet_indices=planet_indices,
        only_earths=only_earths,
        load_disk=load_disk,
    )

    zodi = AYOZodi(
        surface_brightness_mag_arcsec2=zodi_surface_brightness_mag,
    )

    return Scene(system=system, zodi=zodi)
