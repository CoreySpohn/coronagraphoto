# Photon rates, electron rates and observations

`build_exposure(scene, instrument, bands, *, time_d, roll_deg=0.0, plan=None,
order=4)` builds the `Exposure` record: epoch, roll, wavelength quadrature,
optional frame plan, and the ecliptic latitude and solar longitude of the
local-zodi line of sight (defaults 0 and 135 deg; a known limitation, not
derived from the target or the epoch).

`photon_rates(scene, instrument, exposure)` returns `PhotonRates`. Each source field holds photons/s/nm/pixel with shape
`(W,Q,ny,nx)`. Call `rates.integrate()` for band-integrated photons/s/pixel
on the bands carried by the rule.

`expected(scene, instrument, exposure)` returns `ElectronRates`
in electrons/s/pixel, shape `(W,ny,nx)`. It applies the detector's yield at
each wavelength node before accumulation. For small images its result equals
`instrument.detector.electrons(photon_rates(...))`; the streaming
form avoids retaining the node image cube.

`observe(key, scene, instrument, exposure)` requires `exposure.plan`. It calls `detector.draw` on the electron
rates and returns `Image(data, detector_grid, ("band",), Measure.integral(), "electrons")`.
The observation holds the instantaneous scene at `exposure.time_d` throughout the
`FramePlan`. The detector owns source noise, dark current, CIC, excess noise
and read noise. It is called once per observation, so those noise terms are
not multiplied by the number of sources. Reusing a key reproduces the image.

Use deterministic rates for gradients and likelihood means; discrete noisy
draws are not differentiable physical forward models. Access individual
sources through record fields and `.total()` for their sum. A scene with
`zodi=None` has no local zodi. The speckle model is `instrument.speckle`;
construct an instrument without that field to disable an attached model.

## Image orientation and roll

`exposure.roll_deg` is shared by the detector grid, planet placement and disk
rendering (internally `SkyOrientation(roll_deg, parity=-1)`). The roll is the
position angle of detector +y, north through east. At roll zero, +column points west and +row north. At roll 90, north points
right and east up. The shared grid supplies the geometric centre, including
half-integer centres for even arrays.

Planets use the orbit engine's east/north offsets through
`SkyOrientation.sky_to_detector`. The disk is evaluated directly on that sky grid;
there is no additional image rotation or alternative disk PA conversion.
Stored disks keep their native sky geometry. Parametric disks use
`System.midplane_position_angle_deg`, also north through east.

## Spectral and spatial sampling

`build_exposure` builds the quadrature on the host from the instrument and the
scene's zodi, which declare their spectral breaks (the instrument includes an
attached speckle field). The star is left out of the breaks. The rule splits each positive-width band at the declared breaks and
applies its Gauss-Legendre order within each interval. The photon record
carries the quadrature, whose nm weights enter once at integration.

At each node, native coronagraph products are converted with their `Frame`
and resampled with `hwoutils.grids.resample`, honouring each input Image's
reconstruction declaration. Disk radiance uses linear reconstruction and is
multiplied by pixel solid angle in sr once before extended-scene propagation.
Convolution with the band-limited PSF produces a band-limited image. Speckle
samples and noisy detector draws retain the default reconstruction. Cropping
and discretization must be included in an image's numerical error budget.
