# Photon rates, electron rates and observations

`image_rates(system, optical_path, bands, *, time_d, pointing,
quadrature_order=4, zodi=None, speckle=None, ecliptic_latitude_deg=0.0,
solar_longitude_deg=135.0)` returns `PhotonRates`. Each field is an integrand
in photons/s/nm/pixel with shape `(W,Q,ny,nx)`. Integrate it with
`rates.integrate(Quadrature.build(bands, None, quadrature_order))` for photons/s.

`image_electrons` takes exactly the same arguments and returns `ElectronRates`
in electrons/s/pixel, shape `(W,ny,nx)`. It applies the detector's yield at
each wavelength node before accumulation. For small images its result equals
`optical_path.detector.electrons(image_rates(...), quadrature)`; the streaming
form avoids retaining the node image cube.

`observe(key, system, optical_path, bands, plan, *, time_d, pointing, ...)`
accepts the same optional keywords. It calls `detector.draw` on the electron
rates and returns `Image(data, detector_grid, ("band",), "per_pixel", "electrons")`.
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

Bands are positive `hwoutils.spectral.Band` intervals in nm. The reader uses
an unsplit Gauss-Legendre rule; supply band boundaries at sharp spectral
breaks and test convergence in `quadrature_order`. The photon record does not
include integration weights. Their nm unit enters exactly once in integration.

At each node, native coronagraph products are converted with their `Frame`
and resampled with `hwoutils.grids.resample`. PSFs and leakage conserve pixel
flux under covering coarsening; background transmission preserves sampled
amplitude. Cropping and interpolation can lose flux and must be included in
an image's numerical error budget. Disk radiance is multiplied by pixel solid
angle in sr once before extended-scene propagation.
