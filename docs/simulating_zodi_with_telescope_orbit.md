# Simulating zodi with a telescope's orbit

The zodi pipeline in `coronagraphoto.simulation` (specifically
`zodi_rate` and `zodi_readout`) computes the local zodiacal-light
count rate on the detector for one epoch and one line of sight. To
generate a time series of the kind a paper figure or a mission yield
calculation needs, the per-frame geometry has to be threaded in from an
observatory model. This page shows the recommended composition using
`orbix.observatory.ObservatoryL2Halo` for the observatory and
`skyscapes.background.LeinertZodi` for the brightness model, and
ends with a short validation checklist.

## The composition

A single-epoch zodi simulation pulls together four pieces, each
responsible for a distinct part of the calculation. The
`ObservatoryL2Halo` instance provides the heliocentric position of the
telescope and the per-target sky-geometry angles needed for the Leinert
lookup, the `LeinertZodi` instance evaluates the surface
brightness from those angles, the optical path threads the brightness
through the coronagraph's `sky_trans` map and the detector resampling,
and `zodi_rate` returns the per-pixel count rate. Adding
Poisson shot noise on top is the job of `zodi_readout`, which composes a
detector readout step around `zodi_rate`.

The two Leinert inputs are read off the geometry by the observatory
helpers as described in the skyscapes
[Local zodi + telescope geometry][skyscapes-zodi] doc, and then handed
to `zodi_rate`:


```python
from orbix.observatory import ObservatoryL2Halo
from skyscapes.background import LeinertZodi
from coronagraphoto.simulation import zodi_rate

obs = ObservatoryL2Halo.from_default()
zodi = LeinertZodi(reference_mag_arcsec2=22.0)

ecl_lat = float(obs.ecliptic_latitude_deg(mjd, ra_rad, dec_rad))
helio_lon = float(obs.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad))

rate = zodi_rate(
    mjd,
    wavelength_nm=550.0,
    bin_width_nm=50.0,
    zodi=zodi,
    optical_path=optical_path,
    ecliptic_lat_deg=ecl_lat,
    solar_lon_deg=helio_lon,
)
```

The returned `rate` is an `(ny, nx)` array of photons per second on the
detector. Multiplying by an exposure time and passing the result to
`IdealDetector.readout_source_electrons` yields a Poisson-sampled
electron image, and `zodi_readout` provides both steps in one call.

## Year-long simulations

A year-long animation is a simple matter of looping the single-epoch
recipe over a sequence of MJD samples and stepping the geometry along
the L2 halo trajectory. Each frame draws an independent PRNG key so the
Poisson realisations are uncorrelated, and any frame whose Leinert
lookup falls in the out-of-range region returns `NaN`, which downstream
code uses as the "target unobservable this epoch" gate:

```python
import jax
import jax.numpy as jnp
import numpy as np

obs = ObservatoryL2Halo.from_default(equinox_mjd=60575.25)
zodi = LeinertZodi(reference_mag_arcsec2=22.0)
prng_keys = jax.random.split(jax.random.PRNGKey(0), n_frames)
mjds = 60575.25 + np.linspace(0.0, 365.25, n_frames)

ra_rad = jnp.deg2rad(target_ra_deg)
dec_rad = jnp.deg2rad(target_dec_deg)

for i, mjd in enumerate(mjds):
    el = float(obs.ecliptic_latitude_deg(float(mjd), ra_rad, dec_rad))
    sl = float(obs.helio_ecliptic_longitude_deg(float(mjd), ra_rad, dec_rad))
    image = zodi_readout(
        float(mjd), exposure_s, wavelength_nm, bin_width_nm,
        zodi, optical_path, prng_keys[i],
        ecliptic_lat_deg=el, solar_lon_deg=sl,
    )
```

Two reference implementations live in the coronagraphoto-RASTI project
under `hwo-mission-control/burn/coronagraphoto-rasti/notebooks/`.
`zodi_year_animation.py` runs a single target through a full year, with
the L2-halo trajectory and the telescope-to-target pointing vector
overlaid for visual context. `zodi_year_grid_animation.py` runs four
targets at once for visual validation against the expected phase
relationships.

## What changes through the year

The spatial pattern of the zodi image on the detector is constant up to
a scalar, because `zodi_rate` multiplies a uniform sky
brightness by the coronagraph's `sky_trans` map and only the scalar
Leinert factor varies with epoch. The dark-hole region of the detector,
where `sky_trans` is approximately zero, stays dark all year, and the
off-axis regions where `sky_trans` is approximately one all brighten
and dim together as `Δλ_⊙` sweeps. Per-pixel shot noise scales as
`sqrt(rate * t_exp)` of whatever the scalar is at that epoch, so the
relative noise structure is also constant up to the same scalar.

This means the natural "annual quantity" to plot is the scalar
integrated count rate, not any per-pixel metric, because the per-pixel
contrast structure on the detector is fixed. The
`zodi_year_grid_animation.py` notebook uses that integrated count rate
as the time-series axis for exactly this reason.

## Validation checklist

When wiring up a new survey, mission-yield, or paper figure, sanity
checks against the geometry catch the most common mistakes. The
recommended check picks three on-ecliptic targets at ecliptic
longitudes 0°, 90°, and 180°, which in equatorial coordinates are
`(RA=0°, Dec=0°)`, `(RA=90°, Dec=+23.44°)`, and `(RA=180°, Dec=0°)`.
Compensating for obliquity is what dictates the middle target's non-zero
declination, as explained in the
[Local zodi + telescope geometry][skyscapes-zodi] doc. A year-long `zodi_readout` loop over each of those targets and the
`argmax` day of the integrated count rate will show three peaks
separated by roughly 92 days, all of comparable amplitude, because each
target undergoes solar conjunction on a different date but with similar
zodi brightness at conjunction.

A fourth target placed at high ecliptic latitude, for example
`(RA=0°, Dec=+60°)`, serves as the flat-baseline control. The peak
integrated count rate is roughly 50 to 100 times smaller than the
on-ecliptic peaks, and no frame falls into the `NaN` unobservable
window because the line of sight never approaches the Sun. These three
properties are enforced on every build by the integration test in
`coronagraphoto/tests/integration/test_zodi_with_observatory.py`, and
the `zodi_year_grid_animation.py` notebook serves as the visual
companion.

## See also

The skyscapes [Local zodi + telescope geometry][skyscapes-zodi] doc
covers the geometry side of this pipeline.
{class}`orbix.observatory.ObservatoryL2Halo` documents the L2 halo
orbit interpolator and the sky-geometry helpers used above.
{class}`skyscapes.background.LeinertZodi` documents the
Leinert+1998 surface-brightness model.

[skyscapes-zodi]: ../../../skyscapes/docs/local_zodi_geometry.md
