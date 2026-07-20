"""Functions for running full simulations and processing sources.

Public API conventions:

- ``<source>_rate(source, optical_path, *, ...)`` returns the noiseless
  per-pixel photo-electron rate on the detector for one source.
- ``<source>_readout(source, optical_path, prng_key, *, ...)`` returns a
  noisy detector readout (photon Poisson + QE binomial) for one source.
- ``system_rate(scene, optical_path, *, ...)`` sums every per-source rate
  map for a scene (the differentiable forward model).
- ``system_readout(scene, optical_path, prng_key, *, ...)`` sums every
  per-source Poisson-realised readout for a scene.

All observation parameters (``start_time_jd``, ``exposure_time_s``,
``wavelength_nm``, ``bin_width_nm``, ``telescope_pa_deg``,
``ecliptic_lat_deg``, ``solar_lon_deg``) are kwarg-only. The convention
keeps signatures discoverable when more parameters land later (IFS,
multi-roll observations).
"""

import jax
import jax.numpy as jnp
from hwoutils.constants import d2s
from hwoutils.conversions import arcsec_to_lambda_d, lambda_d_to_arcsec
from hwoutils.transforms import ccw_rotation_matrix, resample_flux
from skyscapes.background import Zodi


def pre_coro_bin_processing(flux, bin_center_nm, bin_width_nm, optical_path):
    """Process a bin through the pre-coro elements of the optical path."""
    # ph/s/m^2/nm -> ph/s/m^2
    flux = flux * bin_width_nm
    # ph/s
    flux = flux * optical_path.primary.area_m2
    # apply combined attenuation of mirrors / filters / etc.
    return flux * optical_path.system_throughput(bin_center_nm)


def _detector_sampling_lod(bin_center_nm, optical_path):
    """The detector grid expressed in coronagraph (lambda/D) units.

    The coronagraph contract is sampling-explicit and dimensionless:
    every rate function requests maps directly at the detector grid,
    converted to lambda/D at this wavelength. Chromatic magnification
    is exactly this conversion changing with the bin center.
    """
    pixel_scale_lod = arcsec_to_lambda_d(
        optical_path.detector.pixel_scale_arcsec,
        bin_center_nm,
        optical_path.primary.diameter_m,
    )
    return pixel_scale_lod, optical_path.detector.shape


# ---------------------------------------------------------------------------
# Star
# ---------------------------------------------------------------------------


def star_rate(
    star,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
):
    """Generate the star count rate on the detector."""
    source_diam_lod = arcsec_to_lambda_d(
        star.diameter_arcsec, wavelength_nm, optical_path.primary.diameter_m
    )
    flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)
    pixel_scale_lod, shape = _detector_sampling_lod(wavelength_nm, optical_path)
    image_rate = flux * optical_path.coronagraph.stellar_map(
        wavelength_nm,
        source_diam_lod,
        pixel_scale_lod=pixel_scale_lod,
        shape=shape,
    )
    return jnp.clip(image_rate, 0, None)


def star_readout(
    star,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
):
    """Process a star through the provided optical path."""
    image_rate_detector = star_rate(
        star,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------


def planet_rate(
    planet,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    trig_solver,
):
    """Generate the per-batch planet count rate on the detector.

    Operates on a single ``skyscapes.scene.Planet`` (which internally
    batches K planets sharing the same atmosphere class). The Python
    loop over a heterogeneous ``System.planets`` tuple lives in
    :func:`system_readout`; this function stays inside the per-Planet-type
    JIT cache boundary.
    """
    # The new Planet API takes a 1-D time axis; squeeze T=1.
    source_positions_as = planet.position_arcsec(
        trig_solver, jnp.atleast_1d(start_time_jd), star=star
    )[:, :, 0]  # (2, K)

    # A positive telescope PA corresponds to a CW rotation of the sky.
    rotation_matrix = ccw_rotation_matrix(-telescope_pa_deg)
    source_positions_as = rotation_matrix @ source_positions_as

    source_positions_lod = arcsec_to_lambda_d(
        source_positions_as, wavelength_nm, optical_path.primary.diameter_m
    )

    # ``wavelength_nm`` stays scalar -- the underlying atmosphere reflectivity
    # code expects a scalar and broadcasts internally. ``start_time_jd`` is
    # promoted to (1,) because the orbit propagator needs a T axis.
    flux = planet.spec_flux_density(
        trig_solver,
        wavelength_nm,
        jnp.atleast_1d(start_time_jd),
        star=star,
    )[:, 0]  # (K,) -- drop T=1 axis
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    pixel_scale_lod, shape = _detector_sampling_lod(wavelength_nm, optical_path)
    psfs = optical_path.coronagraph.source_psfs(
        wavelength_nm,
        source_positions_lod[0],
        source_positions_lod[1],
        pixel_scale_lod=pixel_scale_lod,
        shape=shape,
    )
    image_rate = jnp.einsum("i,ijk->jk", flux, psfs)
    return jnp.clip(image_rate, 0, None)


def planet_readout(
    planet,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    trig_solver,
):
    """Process a per-batch Planet through the optical path."""
    image_rate_detector = planet_rate(
        planet,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        telescope_pa_deg=telescope_pa_deg,
        star=star,
        trig_solver=trig_solver,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


def disk_rate(
    disk,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    incl_deg,
    pa_deg,
):
    """Generate the disk count rate on the detector.

    Disks return CONTRAST (dimensionless flux ratio relative to the host
    star); we multiply by ``star.spec_flux_density`` here to convert to
    photon flux density per pixel before resampling and PSF convolution.

    ``incl_deg`` / ``pa_deg`` are the disk's intrinsic orientation in the
    sky frame; ``telescope_pa_deg`` is the telescope's roll. The disk is
    rendered at its intrinsic geometry and the coronagraph's
    ``extended_scene`` rotates it by ``-telescope_pa_deg`` into the
    detector frame while rendering.

    Raises:
        ValueError: from the coronagraph if it cannot render an extended
            scene (e.g. a table-backed coronagraph built without a PSF
            datacube).
    """
    contrast = disk.surface_brightness(wavelength_nm, start_time_jd, incl_deg, pa_deg)
    star_flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = contrast * star_flux
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)

    map_pixel_scale_lod = arcsec_to_lambda_d(
        disk.pixel_scale_arcsec,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )
    pixel_scale_lod, shape = _detector_sampling_lod(wavelength_nm, optical_path)
    image_rate = optical_path.coronagraph.extended_scene(
        flux,
        map_pixel_scale_lod,
        wavelength_nm,
        pixel_scale_lod=pixel_scale_lod,
        shape=shape,
        rotation_deg=-telescope_pa_deg,
    )
    return jnp.clip(image_rate, 0, None)


def disk_readout(
    disk,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    star,
    incl_deg,
    pa_deg,
):
    """Process a disk through the provided optical path.

    ``incl_deg`` / ``pa_deg`` are the disk's intrinsic sky-frame
    orientation; ``system_readout`` pulls them from
    ``scene.system.midplane_inc_deg`` / ``midplane_pa_deg`` so every
    disk component in the System renders at the same midplane.
    """
    image_rate_detector = disk_rate(
        disk,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        telescope_pa_deg=telescope_pa_deg,
        star=star,
        incl_deg=incl_deg,
        pa_deg=pa_deg,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Zodi
# ---------------------------------------------------------------------------


def zodi_rate(
    zodi: Zodi,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Generate the zodi count rate on the detector.

    Treats zodi as a spatially uniform surface-brightness source. The
    coronagraph's sky transmission map sets the per-pixel attenuation;
    no PSF convolution is needed (a flat field convolved with any
    normalised PSF returns itself).
    """
    sb_per_arcsec2 = zodi.spec_flux_density(
        wavelength_nm, start_time_jd, ecliptic_lat_deg, solar_lon_deg
    )
    flux_per_pixel = sb_per_arcsec2 * optical_path.detector.pixel_scale_arcsec**2

    pixel_scale_lod, shape = _detector_sampling_lod(wavelength_nm, optical_path)
    flux_map = flux_per_pixel * optical_path.coronagraph.background_transmission(
        wavelength_nm, pixel_scale_lod=pixel_scale_lod, shape=shape
    )
    flux_map = pre_coro_bin_processing(
        flux_map, wavelength_nm, bin_width_nm, optical_path
    )
    return jnp.clip(flux_map, 0, None)


def zodi_readout(
    zodi: Zodi,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Process a zodi source through the provided optical path."""
    image_rate_detector = zodi_rate(
        zodi,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        ecliptic_lat_deg=ecliptic_lat_deg,
        solar_lon_deg=solar_lon_deg,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Speckle
# ---------------------------------------------------------------------------


def speckle_rate(
    speckle,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    star,
):
    """Generate the speckle count rate on the detector.

    The speckle field returns a CONTRAST delta (fraction of host-star flux
    per pixel) -- the stochastic wavefront-error residual that sits on top
    of the deterministic ``stellar_intens`` floor already applied in
    :func:`star_rate`. We multiply by the host-star flux to convert to a
    photon rate, then resample to the detector. Structurally this mirrors
    :func:`star_rate`, not :func:`disk_rate`: the field is already a
    post-coronagraph focal-plane map, so there is no PSF convolution.

    Evolution is driven by time, not a PRNG key: the elapsed seconds are
    ``(start_time_jd - speckle.epoch_jd)``, so the rate is deterministic
    and differentiable and temporal correlation survives across a roll
    sequence. The realization's randomness is fixed at construction.

    The speckle map is taken on the plane declared by its own
    ``speckle.pixel_scale_lod`` and resampled to the detector grid
    directly, so it need not share a plate scale with the coronagraph.
    """
    time_s = (start_time_jd - speckle.epoch_jd) * d2s
    flux = star.spec_flux_density(wavelength_nm, start_time_jd)
    flux = pre_coro_bin_processing(flux, wavelength_nm, bin_width_nm, optical_path)
    contrast = speckle.realize(wavelength_nm=wavelength_nm, time_s=time_s)
    image_rate_coro = contrast * flux
    speckle_scale_arcsec = lambda_d_to_arcsec(
        speckle.pixel_scale_lod,
        wavelength_nm,
        optical_path.primary.diameter_m,
    )
    image_rate = resample_flux(
        image_rate_coro,
        speckle_scale_arcsec,
        optical_path.detector.pixel_scale_arcsec,
        optical_path.detector.shape,
        0.0,  # speckles are detector-fixed; rotation is applied source-side
    )
    return jnp.clip(image_rate, 0, None)


def speckle_readout(
    speckle,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    star,
):
    """Process a speckle field through the provided optical path.

    The PRNG key is used only for the photon Poisson draw; the speckle
    realization itself is deterministic in time (see :func:`speckle_rate`).
    """
    image_rate_detector = speckle_rate(
        speckle,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
        star=star,
    )
    return optical_path.detector.readout_source_electrons(
        image_rate_detector, exposure_time_s, prng_key
    )


# ---------------------------------------------------------------------------
# Whole-scene orchestrator
# ---------------------------------------------------------------------------


def system_rate(
    scene,
    optical_path,
    *,
    start_time_jd,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Sum of deterministic per-source count rates for a :class:`~skyscapes.Scene`.

    The differentiable companion to :func:`system_readout`. Returns the
    total rate map (electrons/s/pixel, no Poisson noise, no QE multiply)
    summing star, every planet, the optional disk, the optional zodi, and
    the optional speckle field on ``optical_path``. Use this for likelihood
    evaluation, retrievals, or any inference loop that needs gradients
    through the full forward model.
    """
    has_disk = scene.system.disk is not None
    has_zodi = scene.zodi is not None

    total = star_rate(
        scene.system.star,
        optical_path,
        start_time_jd=start_time_jd,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )

    for planet in scene.system.planets:
        total = total + planet_rate(
            planet,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            trig_solver=scene.system.trig_solver,
        )

    if has_disk:
        total = total + disk_rate(
            scene.system.disk,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            incl_deg=jnp.asarray(scene.system.midplane_inc_deg),
            pa_deg=jnp.asarray(scene.system.midplane_pa_deg),
        )

    if has_zodi:
        total = total + zodi_rate(
            scene.zodi,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            ecliptic_lat_deg=ecliptic_lat_deg,
            solar_lon_deg=solar_lon_deg,
        )

    if optical_path.speckle is not None:
        total = total + speckle_rate(
            optical_path.speckle,
            optical_path,
            start_time_jd=start_time_jd,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            star=scene.system.star,
        )

    return total


def system_readout(
    scene,
    optical_path,
    prng_key,
    *,
    start_time_jd,
    exposure_time_s,
    wavelength_nm,
    bin_width_nm,
    telescope_pa_deg,
    ecliptic_lat_deg,
    solar_lon_deg,
):
    """Simulate a full :class:`~skyscapes.Scene` through the optical path.

    Sums per-source detector readouts. Each source consumes its own
    independent PRNG subkey (see :mod:`jax.random` best practices). The
    optional speckle field on ``optical_path`` is the last source and
    consumes the final subkey, so scenes run without one are unaffected.

    The Python loop over ``scene.system.planets`` is intentionally
    unjitted -- it orchestrates JIT-cached per-Planet-type kernels. The
    expensive math is inside each ``planet_readout`` call, not the loop.
    """
    has_disk = scene.system.disk is not None
    has_zodi = scene.zodi is not None
    has_speckle = optical_path.speckle is not None

    n_keys = (
        1 + len(scene.system.planets) + int(has_disk) + int(has_zodi) + int(has_speckle)
    )
    keys = iter(jax.random.split(prng_key, n_keys))

    total = star_readout(
        scene.system.star,
        optical_path,
        next(keys),
        start_time_jd=start_time_jd,
        exposure_time_s=exposure_time_s,
        wavelength_nm=wavelength_nm,
        bin_width_nm=bin_width_nm,
    )

    for planet in scene.system.planets:
        total = total + planet_readout(
            planet,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            trig_solver=scene.system.trig_solver,
        )

    if has_disk:
        total = total + disk_readout(
            scene.system.disk,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            telescope_pa_deg=telescope_pa_deg,
            star=scene.system.star,
            incl_deg=jnp.asarray(scene.system.midplane_inc_deg),
            pa_deg=jnp.asarray(scene.system.midplane_pa_deg),
        )

    if has_zodi:
        total = total + zodi_readout(
            scene.zodi,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            ecliptic_lat_deg=ecliptic_lat_deg,
            solar_lon_deg=solar_lon_deg,
        )

    if has_speckle:
        total = total + speckle_readout(
            optical_path.speckle,
            optical_path,
            next(keys),
            start_time_jd=start_time_jd,
            exposure_time_s=exposure_time_s,
            wavelength_nm=wavelength_nm,
            bin_width_nm=bin_width_nm,
            star=scene.system.star,
        )

    return total
