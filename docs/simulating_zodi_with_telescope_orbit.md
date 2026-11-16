# Local zodi from telescope geometry

The observer supplies ecliptic latitude and heliocentric ecliptic longitude.
`LeinertZodi.photon_radiance` returns photons/s/m2/nm/sr; the image reader
multiplies by collecting area, optical throughput, detector pixel solid angle
and the coronagraph's background transmission. QE belongs to the detector.

For an existing system and optical path:

```python notest
from hwoutils.grids import SkyOrientation
from hwoutils.spectral import Band
from orbix.observatory import ObservatoryL2Halo
from skyscapes.background import LeinertZodi
from coronagraphoto import image_electrons

observer = ObservatoryL2Halo.from_default(equinox_mjd=60575.25)
lat = observer.ecliptic_latitude_deg(mjd, ra_rad, dec_rad)
lon = observer.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad)
rates = image_electrons(
    system,
    optical_path,
    (Band(500.0, 600.0),),
    time_d=time_d,
    pointing=SkyOrientation(0.0),
    zodi=LeinertZodi(),
    ecliptic_latitude_deg=lat,
    solar_longitude_deg=lon,
)
local_zodi_e_per_s = rates.local_zodi
```

The observatory's MJD is an absolute date. `time_d` is days from the system's
scene epoch; convert explicitly on the host. AYO and precomputed backgrounds
accept the same geometry keywords but do not depend on them. For a noisy
whole-system image, pass the same keywords to `observe` with a PRNG key and
`FramePlan`. Sun keepout is an observing constraint, independent of the
background radiance calculation.
