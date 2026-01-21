"""ExoVista data loader utilities using refactored source objects."""

import warnings
from typing import Optional, Sequence

import interpax
import jax.numpy as jnp
from astropy.io.fits import getdata, getheader
from orbix.equations.orbit import mean_anomaly_tp
from orbix.system.planets import Planets as OrbixPlanet

from coronagraphoto import constants as const
from coronagraphoto import conversions as conv
from coronagraphoto.core.sky_scene import SkyScene
from coronagraphoto.core.sources import (
    DiskSource,
    PlanetSources,
    StarSource,
    ZodiSource,
)
from coronagraphoto.transforms.orbital_mechanics import (
    state_vector_to_keplerian,
)


def load_star_from_exovista(fits_file: str, fits_ext: int = 4) -> StarSource:
    """Load star data from ExoVista FITS and create a StarSource.

    Args:
        fits_file: Path to the ExoVista FITS file.
        fits_ext: FITS extension for star data (default: 4).

    Returns:
        StarSource object.
    """
    with open(fits_file, "rb") as f:
        obj_data, obj_header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)

    # Convert wavelengths to nm
    wavelengths_nm = jnp.asarray(conv.um_to_nm(wavelengths_um))

    # Times in decimal years
    times_year = jnp.asarray(2000.0 + obj_data[:, 0])

    # Convert times to julian days
    times_jd = conv.decimal_year_to_jd(times_year)

    # Star flux density in Janskys
    # Shape: (n_wavelengths, n_times)
    flux_density_jy = jnp.asarray(obj_data[:, 16:].T.astype(jnp.float32))

    # Angular diameter in arcseconds
    diameter_arcsec = obj_header["ANGDIAM"] * 1e-3  # mas to arcsec

    # Mass in solar masses
    mass_solar = obj_header.get("MASS")
    mass_kg = conv.Msun_to_kg(mass_solar)

    # Distance in parsecs
    dist_pc = obj_header.get("DIST")

    # Midplane information
    midplane_pa = obj_header.get("PA", 0.0)  # Position angle in degrees
    midplane_i = obj_header.get("I", 0.0)  # Inclination in degrees

    return StarSource(
        diameter_arcsec=diameter_arcsec,
        mass_kg=mass_kg,
        dist_pc=dist_pc,
        midplane_pa_deg=midplane_pa,
        midplane_i_deg=midplane_i,
        wavelengths_nm=wavelengths_nm,
        times_jd=times_jd,
        flux_density_jy=flux_density_jy,
    )


def load_planets_from_exovista(
    fits_file: str,
    star: StarSource,
    planet_indices: Sequence[int],
    required_planets: Optional[int] = None,
) -> PlanetSources:
    """Load specific planet data from ExoVista FITS and create a PlanetSources.

    Args:
        fits_file: Path to the ExoVista FITS file.
        star: The host star source object.
        planet_indices: List of indices of planets to load (0-based).
        required_planets: Optional maximum number of planets to load. If the number
            of planets loaded is less than this, the output will be padded with
            "ghost" planets (invisible, massless, trivial orbits).

    Returns:
        PlanetSources object containing the specified planets.
    """
    planet_ext_start = 5  # ExoVista hardcodes this
    oe_params = {
        "a": [],
        "e": [],
        "i": [],
        "W": [],
        "w": [],
        "M0": [],
        "mass": [],
        "radius": [],
        "p": [],
    }
    contrast_grids = []

    with open(fits_file, "rb") as f:
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)
    wavelengths_nm = jnp.asarray(conv.um_to_nm(wavelengths_um))

    t0 = None  # Will be set from the first planet

    for i in planet_indices:
        with open(fits_file, "rb") as f:
            obj_data, obj_header = getdata(
                f, ext=planet_ext_start + i, header=True, memmap=False
            )

        times_year = jnp.asarray(2000.0 + obj_data[:, 0])
        times_jd = conv.decimal_year_to_jd(times_year)
        if t0 is None:
            t0 = times_jd[0]

        contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(jnp.float32))

        # State vectors to orbital elements
        planet_r_sky_au = obj_data[0, 9:12]
        planet_v_sky_au_yr = obj_data[0, 12:15]
        planet_r_sky = jnp.array(conv.au_to_m(planet_r_sky_au))
        planet_v_sky = jnp.array(conv.au_per_yr_to_m_per_s(planet_v_sky_au_yr))
        mass_earth = obj_header.get("M")
        planet_mass_kg = conv.Mearth_to_kg(float(mass_earth))
        total_mass_kg = star.mass_kg + planet_mass_kg
        mu = const.G_kg_m_s * total_mass_kg
        _a, _e, i_rad, W_rad, w_rad, M_rad = state_vector_to_keplerian(
            planet_r_sky, planet_v_sky, mu
        )

        # Append orbital elements
        oe_params["a"].append(obj_header.get("A"))
        oe_params["e"].append(obj_header.get("E"))
        oe_params["i"].append(jnp.degrees(i_rad))
        oe_params["W"].append(jnp.degrees(W_rad))
        oe_params["w"].append(jnp.degrees(w_rad))
        oe_params["M0"].append(jnp.degrees(M_rad))
        oe_params["mass"].append(obj_header.get("M"))
        oe_params["radius"].append(obj_header.get("R"))
        oe_params["p"].append(obj_header.get("p", 0.2))

        # Create temporary OrbixPlanet to calculate mean anomaly for gridding
        temp_planet = OrbixPlanet(
            Ms=jnp.atleast_1d(star.mass_kg),
            dist=jnp.atleast_1d(star.dist_pc),
            a=jnp.atleast_1d(oe_params["a"][-1]),
            e=jnp.atleast_1d(oe_params["e"][-1]),
            W=jnp.atleast_1d(jnp.deg2rad(oe_params["W"][-1])),
            i=jnp.atleast_1d(jnp.deg2rad(oe_params["i"][-1])),
            w=jnp.atleast_1d(jnp.deg2rad(oe_params["w"][-1])),
            M0=jnp.atleast_1d(jnp.deg2rad(oe_params["M0"][-1])),
            t0=jnp.atleast_1d(t0),
            # orbix Planets expects Mp in Earth masses and Rp in Earth radii
            # (it converts internally to kg and AU respectively)
            Mp=jnp.atleast_1d(oe_params["mass"][-1]),
            Rp=jnp.atleast_1d(oe_params["radius"][-1]),
            p=jnp.atleast_1d(oe_params["p"][-1]),
        )
        mean_anomaly_coords = jnp.rad2deg(
            mean_anomaly_tp(times_jd, temp_planet.n, temp_planet.tp) % (2 * jnp.pi)
        )

        # Resample contrast data onto a regular mean anomaly grid
        # Sort by mean anomaly to ensure valid grid for interpolation
        sort_idx = jnp.argsort(mean_anomaly_coords)
        mean_anomaly_coords_sorted = mean_anomaly_coords[sort_idx]
        contrast_data_sorted = contrast_data[:, sort_idx]

        mean_anomaly_grid = jnp.linspace(0, 360, 100)
        xq, yq = jnp.meshgrid(wavelengths_nm, mean_anomaly_grid, indexing="ij")
        contrast_grid = interpax.interp2d(
            xq.flatten(),
            yq.flatten(),
            wavelengths_nm,
            mean_anomaly_coords_sorted,
            contrast_data_sorted,
            method="linear",
            extrap=True,
        ).reshape(xq.shape)
        contrast_grids.append(contrast_grid)

    n_planets_loaded = len(planet_indices)

    # Pad with ghost planets if necessary, or truncate if too many
    # Useful to keep the shape constant to avoid recompilation
    if required_planets is not None:
        if n_planets_loaded > required_planets:
            warnings.warn(
                f"Loaded {n_planets_loaded} planets, but required_planets is {required_planets}. "
                f"Truncating to first {required_planets} planets.",
                UserWarning,
                stacklevel=2,
            )
            # Slice all data structures to only include first required_planets
            for key in oe_params:
                oe_params[key] = oe_params[key][:required_planets]
            contrast_grids = contrast_grids[:required_planets]
            n_planets_loaded = required_planets

        n_ghosts = required_planets - n_planets_loaded
        if n_ghosts > 0:
            # Add trivial orbital parameters for ghosts
            oe_params["a"].extend([1.0] * n_ghosts)
            oe_params["e"].extend([0.0] * n_ghosts)
            oe_params["i"].extend([0.0] * n_ghosts)
            oe_params["W"].extend([0.0] * n_ghosts)
            oe_params["w"].extend([0.0] * n_ghosts)
            oe_params["M0"].extend([0.0] * n_ghosts)
            oe_params["mass"].extend([0.0] * n_ghosts)
            oe_params["radius"].extend([0.0] * n_ghosts)
            oe_params["p"].extend([0.0] * n_ghosts)

            # Add zero-contrast grids
            # Shape of one grid: (n_wavelengths, n_mean_anomalies)
            # We use the shape of the first real planet's grid (or a default if no real planets)
            if n_planets_loaded > 0:
                base_shape = contrast_grids[0].shape
            else:
                # Fallback if trying to load 0 real planets but some ghosts (unlikely but possible)
                base_shape = (len(wavelengths_nm), 100)

            zero_grid = jnp.zeros(base_shape, dtype=jnp.float32)
            contrast_grids.extend([zero_grid] * n_ghosts)

    # Update count to include ghosts
    n_total_planets = len(oe_params["a"])

    # Create a single OrbixPlanets object for all planets
    orbix_planets = OrbixPlanet(
        Ms=jnp.atleast_1d(star.mass_kg),
        dist=jnp.atleast_1d(star.dist_pc),
        a=jnp.array(oe_params["a"]),
        e=jnp.array(oe_params["e"]),
        W=jnp.deg2rad(jnp.array(oe_params["W"])),
        i=jnp.deg2rad(jnp.array(oe_params["i"])),
        w=jnp.deg2rad(jnp.array(oe_params["w"])),
        M0=jnp.deg2rad(jnp.array(oe_params["M0"])),
        t0=jnp.repeat(t0, n_total_planets),
        # orbix Planets expects Mp in Earth masses and Rp in Earth radii
        # (it converts internally to kg and AU respectively)
        Mp=jnp.array(oe_params["mass"]),
        Rp=jnp.array(oe_params["radius"]),
        p=jnp.array(oe_params["p"]),
    )

    # Stack contrast grids and create interpolator
    # If only one planet is loaded (and no ghosts), pad with a duplicate
    if n_total_planets == 1:
        stacked_contrast_grid = jnp.stack(contrast_grids * 2, axis=-1)
        contrast_interp_indices = jnp.array([0, 1])
    else:
        stacked_contrast_grid = jnp.stack(contrast_grids, axis=-1)
        contrast_interp_indices = jnp.arange(n_total_planets)

    contrast_interp = interpax.Interpolator3D(
        wavelengths_nm,
        mean_anomaly_grid,
        contrast_interp_indices,
        stacked_contrast_grid,
        method="linear",
    )

    return PlanetSources(
        star=star,
        contrast_interp=contrast_interp,
        orbix_planets=orbix_planets,
    )


def load_disk_from_exovista(
    fits_file: str, fits_ext: int, star: StarSource
) -> DiskSource:
    """Load disk data from ExoVista FITS and create a DiskSource.

    Args:
        fits_file: Path to the ExoVista FITS file.
        fits_ext: FITS extension for disk data.
        star: The host star source object.

    Returns:
        DiskSource object.
    """
    with open(fits_file, "rb") as f:
        obj_data, header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=fits_ext - 1, header=False, memmap=False)

    # Convert wavelengths to nm
    wavelengths_nm = jnp.asarray(conv.um_to_nm(wavelengths_um))

    # Disk contrast cube, removing the last frame (numerical noise estimate)
    # Shape: (n_wavelengths, ny, nx)
    contrast_cube = jnp.asarray(obj_data[:-1].astype(jnp.float32))

    # Pixel scale in arcsec/pixel
    pixel_scale_arcsec = conv.mas_to_arcsec(header["PXSCLMAS"])

    return DiskSource(
        pixel_scale_arcsec=pixel_scale_arcsec,
        star=star,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
    )


def get_earth_like_planet_indices(fits_file: str) -> list[int]:
    """Identify Earth-like planets in an ExoVista FITS file.

    Uses the same classification criteria as exoverses:
    - Scaled semi-major axis: 0.95 ≤ a / sqrt(L_star) < 1.67 AU
    - Planet radius: 0.8 / sqrt(a_scaled) ≤ R < 1.4 Earth radii

    Args:
        fits_file:
            Path to the ExoVista FITS file.

    Returns:
        List of planet indices (0-based) that are Earth-like.
    """
    import numpy as np

    # Get the number of planets and star luminosity
    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
        _, star_header = getdata(f, ext=4, header=True, memmap=False)

    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4
    star_luminosity_lsun = star_header.get("LSTAR", 1.0)

    planet_ext_start = 5
    earth_indices = []

    for i in range(n_planets_total):
        with open(fits_file, "rb") as f:
            _, planet_header = getdata(
                f, ext=planet_ext_start + i, header=True, memmap=False
            )

        # Get orbital and physical parameters
        a_au = planet_header.get("A", 1.0)  # Semi-major axis in AU
        radius_rearth = planet_header.get("R", 1.0)  # Radius in Earth radii

        # Scaled semi-major axis (reverse luminosity scaling)
        a_scaled = a_au / np.sqrt(star_luminosity_lsun)

        # Earth-like classification criteria (from exoverses)
        lower_a = 0.95
        upper_a = 1.67
        lower_r = 0.8 / np.sqrt(a_scaled)
        upper_r = 1.4

        is_earth = (lower_a <= a_scaled < upper_a) and (
            lower_r <= radius_rearth < upper_r
        )

        if is_earth:
            earth_indices.append(i)

    return earth_indices


def load_sky_scene_from_exovista(
    fits_file: str,
    planet_indices: Optional[Sequence[int]] = None,
    required_planets: Optional[int] = None,
    only_earths: bool = False,
) -> SkyScene:
    """Load complete sky scene from ExoVista FITS file.

    Args:
        fits_file:
            Path to the ExoVista FITS file.
        planet_indices:
            Optional list of planet indices to load. If None, loads all planets
            (or only Earths if only_earths=True).
        required_planets:
            Optional required number of planets to load. Passed to
            load_planets_from_exovista to ensure fixed array sizes.
        only_earths:
            If True, automatically filter to only include Earth-like planets.
            Uses the same classification criteria as exoverses. This parameter
            is ignored if planet_indices is explicitly provided.

    Returns:
        SkyScene object containing all sources.
    """
    # fits file extensions, exoVista hard codes these
    disk_ext = 2

    # Get the number of planets
    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4

    if planet_indices is None:
        if only_earths:
            planet_indices = get_earth_like_planet_indices(fits_file)
        else:
            planet_indices = range(n_planets_total)

    # Load the star (always at extension 4)
    star = load_star_from_exovista(fits_file, fits_ext=4)

    # Create ZodiSource with flat 23 mag/arcsec^2
    wavelengths_nm = star._wavelengths_nm
    zodi_mag_per_arcsec2 = 23.0
    zodi_flux_jy_arcsec2 = conv.mag_per_arcsec2_to_jy_per_arcsec2(zodi_mag_per_arcsec2)
    # Broadcast to wavelengths shape
    flux_density_jy_arcsec2 = jnp.full_like(wavelengths_nm, zodi_flux_jy_arcsec2)
    zodi = ZodiSource(
        wavelengths_nm=wavelengths_nm,
        flux_density_jy_arcsec2=flux_density_jy_arcsec2,
    )

    # Load the planets
    planets = load_planets_from_exovista(
        fits_file, star, planet_indices, required_planets=required_planets
    )

    # Load disk
    disk = load_disk_from_exovista(fits_file, disk_ext, star)

    return SkyScene(stars=star, planets=planets, disk=disk, zodi=zodi)
