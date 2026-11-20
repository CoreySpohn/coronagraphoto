# Local zodi from telescope geometry

The image path gives the local-zodi model the scene's sky frame (RA, Dec and
epoch) and the exposure's `time_d`. `LeinertZodi` derives the ecliptic latitude
and the Sun-relative longitude of the target from them
(`orbix.observatory.ecliptic_pointing`).
`LeinertZodi.photon_radiance(wavelength_nm, sky_frame, time_d)` returns photons/s/m2/nm/sr; the image reader
multiplies by collecting area, optical throughput, detector pixel solid angle
and the coronagraph's background transmission. QE belongs to the detector.

For an existing scene and instrument:

```python notest
import equinox as eqx
from optixstuff.spectral import Band
from skyscapes import LeinertZodi
from coronagraphoto import build_exposure, expected

zodi = LeinertZodi()
# scene is a skyscapes.Scene, for example from from_exovista
scene = eqx.tree_at(lambda s: s.local_zodi, scene, zodi, is_leaf=lambda x: x is None)
exposure = build_exposure(scene, instrument, (Band(500.0, 600.0),), time_d=time_d)
rates = expected(scene, instrument, exposure)
local_zodi_e_per_s = rates.local_zodi
```

`time_d` is days from the system's scene epoch; convert absolute dates
explicitly on the host with `scene.epoch.days(...)`. The sightline follows the
Sun through the year, so the local zodi changes with `time_d`. AYO and precomputed backgrounds
accept the same frame and time but do not depend on them. For a noisy
whole-system image, build the exposure with a `FramePlan` and pass it to
`observe` with a PRNG key. Sun keepout is an observing constraint, independent of the
background radiance calculation.
