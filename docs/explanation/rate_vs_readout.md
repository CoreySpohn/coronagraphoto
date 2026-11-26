# Photon rates, electron rates and observations

`build_exposure(scene, instrument, bands, *, time_d, roll_deg=0.0, offset_mas=None,
plan=None, order=4, field_lod=None, allow_wide_bands=False)` builds the
`Exposure` record with observation time, pointing, wavelength quadrature,
the `Resolution` that reduces each tabulated spectrum of the scene to the
nodes and the optional frame plan. The star, planet and local-zodi values are not stored: the image path
asks `scene.sources` for them at each node, given the scene's sky frame and
`time_d`. `field_lod` is the largest separation, in lambda/D at the band centre,
at which the response is read. The default, None, takes the field the instrument
declares, its detector half-diagonal; the nodes of a piece grow with the field.
A single-wavelength coronagraph response scales with the node wavelength. A
band-averaged product already includes chromatic blur. Its star, planet, sky
and disk response uses the same angular scale at every node of the observing
band, set by that band's centre. Additional nodes sample source spectra,
instrument efficiency and the speckle field. If the observing band is wider
than the product, the rendered image contains only the product's blur. The build refuses
fractional widths above 1.25 times the product's fractional width and gives
accepted tiles. Passing `allow_wide_bands=True` accepts the product's blur at
one scale, following the count-rate recipe of Stark and Krist (2019),
"Standardized Coronagraph Parameters for Input into Yield Calculations".

A caller that reads a smaller field passes it and renders
fewer nodes; with 0 each piece holds exactly `order` nodes. The image routes
render the real nodes of each band only, so a band with fewer nodes than the
widest costs no padded renders.

`photon_rates(scene, instrument, exposure)` returns photon `Rates`. Each source field holds photons/s/nm/pixel with shape
`(W,Q,ny,nx)`. Call `rates.integrate()` for band-integrated photons/s/pixel
on the bands carried by the rule.

`expected(scene, instrument, exposure)` returns electron `Rates`
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
`local_zodi=None` has no local zodi. The speckle model is `instrument.speckle`;
construct an instrument without that field to disable an attached model. The
render converts the exposure time with the field's `elapsed_s(time_d)`; under
x32 a field with a nonzero `epoch_d` is refused, and the ways out are to enable
x64, to move the scene epoch to the observation so that `epoch_d` is zero, or to
anchor the field at zero.

## Precision

For an AB magnitude 5 star and a planet at 0.05 arcsec on a 48 × 48 detector
at zero roll (1600 nodes over 500–600 nm), the float32 `expected` image agreed
with float64 to 1.46 × 10⁻⁶ of the float64 image peak at the worst pixel
and the detector-summed planet electron rate agreed to 5.11 × 10⁻⁷ of its
float64 value, so use float64 when you need finer agreement.

## Image orientation and roll

`exposure.pointing.roll_deg` is shared by the detector grid, planet placement
and disk rendering. The roll is the position angle of detector +y, north
through east. At roll zero, +column points west and +row north. At roll 90,
north points right and east up. The grid centre is the optical axis, including
half-integer centres for even arrays.

Planets use the orbit engine's east/north offsets through
`Pointing.detector_offsets_arcsec` with parity -1. Rendering requires
`offset_mas=None`, which places the target on the optical axis. The disk is
evaluated directly on that sky grid;
there is no additional image rotation or alternative disk PA conversion.
Stored disks keep their native sky geometry. Parametric disks use
`Scene.midplane_position_angle_deg`, also north through east.

## Spectral and spatial sampling

`build_exposure` builds the quadrature on the host from the instrument and the
scene's tabulated sources, which declare their spectral breaks (the instrument includes an
attached speckle field). The tables resolve through the exposure's `Resolution`
rather than adding a node per knot, so a rough star adds no nodes. The rule splits each positive-width band at the declared breaks and
applies its Gauss-Legendre order within each interval. The photon record
carries the quadrature, whose nm weights enter once at integration.

Spectral breaks come from the instrument, the local zodi and the disk. A disk
has no reduction operator: its radiance is read at the nodes. A disk with a
tabulated spectrum, such as the contrast cube of an ExoVista disk, therefore
declares its wavelength knots as breaks, and each knot interval inside a band
gets its own nodes, so a narrow feature in the disk spectrum is integrated
rather than missed between nodes. This costs one set of nodes per knot
interval; a disk that declares no breaks (an analytic exozodi law) adds none.

`planet_source_support(scene, instrument, exposure)` returns a `(W, K, T)`
boolean: whether each planet at each time lies inside the source domain of the
coronagraph component the renderer selects for each band, read at the band
center. For single-wavelength products a planet's separation in lambda/D
changes across the band, and nodes on opposite sides of a component boundary
read different components. For band averages one child covers the entire
observing band and the planet keeps its band-centre separation.

At each node, native coronagraph products are converted with their `Frame`
and resampled with `optixstuff.grids.resample`, honouring each input Image's
reconstruction declaration. Disk radiance uses linear reconstruction and is
multiplied by pixel solid angle in sr once before extended-scene propagation.
Convolution with the band-limited PSF produces a band-limited image. Speckle
samples and noisy detector draws retain the default reconstruction. Cropping
and discretization must be included in an image's numerical error budget.
