# Photon rates, electron rates and observations

`build_exposure(scene, instrument, bands, *, time_d, roll_deg=0.0, plan=None,
order=4, field_lod=None)` builds the `Exposure` record: observation time, roll,
wavelength quadrature, the `Resolution` that reduces each tabulated spectrum of
the scene to the nodes, the knot leaves the build read (`reads`) and the optional
frame plan. The star, planet and local-zodi values are not stored: the image path
asks `scene.sources` for them at each node, given the scene's sky frame and
`time_d`. `field_lod` is the largest separation, in lambda/D at the band centre,
at which the response is read. The default, None, takes the field the instrument
declares, its detector half-diagonal; the nodes of a piece grow with the field.
The field is measured from the axis and sets the nodes for a response that is
smooth in wavelength. The per-pixel accuracy of a product read from a table is
set by how the detector pixels read the table instead: the reconstruction the
product declares, and whether pixels sample it or table cells are deposited
onto them, which changes where a pixel equals a table cell. It depends on the
package and the pixel scale, and a wider field does not usefully reduce it. One
measured case is the `eac1_aavc_2d` package with 0.25 lambda/D pixels at the
centre of a 20 percent band, on its 0.25 lambda/D tables: the star leakage map
is good to 2.9e-3 of its peak per pixel at the default field (1.9e-3 at twice
the field), to 1e-3 in a 0.7 lambda/D aperture and to 1.3e-4 on the detector
sum. Other measured cases run from 1e-8 (a smooth test table at 0.45 lambda/D)
to 1.4e-2 (the same package at 0.40 lambda/D). The image of an off-axis source
slides about the source, which the default field does not count; on that case,
with a half-diagonal of 22.6 lambda/D, planets at 3 to 20 lambda/D are held to
3.4e-5 of their own peak. A caller that reads a smaller field passes it and renders
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
`Scene.midplane_position_angle_deg`, also north through east.

## Spectral and spatial sampling

`build_exposure` builds the quadrature on the host from the instrument and the
scene's tabulated sources, which declare their spectral breaks (the instrument includes an
attached speckle field). The tables resolve through the exposure's `Resolution`
rather than adding a node per knot, so a rough star adds no nodes. The rule splits each positive-width band at the declared breaks and
applies its Gauss-Legendre order within each interval. The photon record
carries the quadrature, whose nm weights enter once at integration.

At each node, native coronagraph products are converted with their `Frame`
and resampled with `optixstuff.grids.resample`, honouring each input Image's
reconstruction declaration. Disk radiance uses linear reconstruction and is
multiplied by pixel solid angle in sr once before extended-scene propagation.
Convolution with the band-limited PSF produces a band-limited image. Speckle
samples and noisy detector draws retain the default reconstruction. Cropping
and discretization must be included in an image's numerical error budget.
