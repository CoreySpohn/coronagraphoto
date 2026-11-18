# Local zodi from telescope geometry

The observer supplies ecliptic latitude and heliocentric ecliptic longitude.
`LeinertZodi.photon_radiance` returns photons/s/m2/nm/sr; the image reader
multiplies by collecting area, optical throughput, detector pixel solid angle
and the coronagraph's background transmission. QE belongs to the detector.

For an existing scene and instrument:

```python notest
from dataclasses import replace

from hwoutils.spectral import Band
from orbix.observatory import ObservatoryL2Halo
from zodi.jax import LeinertZodi
from skyscapes import Scene
from coronagraphoto import build_exposure, expected

observer = ObservatoryL2Halo.from_default(equinox_mjd=60575.25)
lat = observer.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)
lon = observer.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)
zodi = LeinertZodi()
scene = Scene(system, zodi)
exposure = build_exposure(scene, instrument, (Band(500.0, 600.0),), time_d=time_d)
exposure = replace(exposure, ecliptic_latitude_deg=lat, solar_longitude_deg=lon)
rates = expected(scene, instrument, exposure)
local_zodi_e_per_s = rates.local_zodi
```

The observatory's MJD is an absolute date. `time_d` is days from the system's
scene epoch; convert explicitly on the host. AYO and precomputed backgrounds
accept the same geometry fields but do not depend on them. For a noisy
whole-system image, build the exposure with a `FramePlan` and pass it to
`observe` with a PRNG key. Sun keepout is an observing constraint, independent of the
background radiance calculation.
