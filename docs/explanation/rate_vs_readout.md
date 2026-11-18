# Photon rates, electron rates and observations

`image_rates(system, optical_path, quadrature, *, time_d, pointing,
zodi=None, speckle=None, ecliptic_latitude_deg=0.0, solar_longitude_deg=135.0)`
returns `PhotonRates`. Each source field holds photons/s/nm/pixel with shape
`(W,Q,ny,nx)`. Call `rates.integrate()` for band-integrated photons/s/pixel
on the bands carried by the rule.

`image_electrons` takes exactly the same arguments and returns `ElectronRates`
in electrons/s/pixel, shape `(W,ny,nx)`. It applies the detector's yield at
each wavelength node before accumulation. For small images its result equals
`optical_path.detector.electrons(image_rates(...))`; the streaming
form avoids retaining the node image cube.

`observe(key, system, optical_path, quadrature, plan, *, time_d, pointing, ...)`
accepts the same optional keywords. It calls `detector.draw` on the electron
rates and returns `Image(data, detector_grid, ("band",), Measure.integral(), "electrons")`.
The observation holds the instantaneous scene at `time_d` throughout the
`FramePlan`. The detector owns source noise, dark current, CIC, excess noise
and read noise. It is called once per observation, so those noise terms are
not multiplied by the number of sources. Reusing a key reproduces the image.

Use deterministic rates for gradients and likelihood means; discrete noisy
draws are not differentiable physical forward models. Access individual
sources through record fields and `.total()` for their sum. `zodi=None`
disables local zodi. `speckle=None` uses `optical_path.speckle`; construct a
path without that field to disable an attached speckle model.

## Image orientation and roll

`pointing=SkyOrientation(pa_y_deg, parity=-1)` is shared by the detector grid,
planet placement and disk rendering. PA is north through east for detector
+y. At PA zero, +column points west and +row north. At PA 90, north points
right and east up. The shared grid supplies the geometric centre, including
half-integer centres for even arrays.

Planets use the orbit engine's east/north offsets through
`pointing.sky_to_detector`. The disk is evaluated directly on that sky grid;
there is no additional image rotation or alternative disk PA conversion.
Stored disks keep their native sky geometry. Parametric disks use
`System.midplane_position_angle_deg`, also north through east.

## Spectral and spatial sampling

Use `Quadrature.build(bands, system, optical_path, zodi)` on the host when
local zodi is active. The system declares the star, planets and attached disk.
Include an explicitly supplied speckle field when it contributes. Omit absent
models. The rule splits each positive-width band at the declared breaks and
applies its Gauss-Legendre order within each interval. The photon record
carries the quadrature, whose nm weights enter once at integration.

At each node, native coronagraph products are converted with their `Frame`
and resampled with `hwoutils.grids.resample`, honouring each input Image's
reconstruction declaration. Disk radiance uses linear reconstruction and is
multiplied by pixel solid angle in sr once before extended-scene propagation.
Convolution with the band-limited PSF produces a band-limited image. Speckle
samples and noisy detector draws retain the default reconstruction. Cropping
and discretization must be included in an image's numerical error budget.
