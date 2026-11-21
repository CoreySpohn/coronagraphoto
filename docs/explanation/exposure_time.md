# Aperture rates, S/N and exposure time

The image readers answer "what does the detector see". A yield or scheduling
code asks a narrower question many millions of times: how long until this
planet reaches a signal-to-noise ratio. Rendering an image per question is far
too slow, so the answer is read from tables that the image path fills once.

## Three steps, three cadences

| Step | Function | When | Cost here |
|---|---|---|---|
| Response | `aperture_response(instrument, quadrature, rule=...)` | once per instrument and band set | seconds to minutes |
| Record | `aperture_record(scene, instrument, exposure, response)` | once per target, instrument and band set | one background render |
| Read | `aperture_rates(record, east_arcsec, north_arcsec, contrast, time_d=...)` | once per planet and observation time | a few gathers |
| Scene read | `scene_aperture_rates(record, scene, time_d, roll_deg=...)` | once per scene and set of times | a vmap of the read |

**Response.** Point sources are placed on rings around the star, imaged on
the detector pixels by the coronagraph, and summed in a circular aperture
centred on each source. The ring mean is the planet throughput at that
separation. The response also holds the aperture's pixel weights for a
centre at each half-pixel phase. Nothing in it depends on a target.

**Record.** The star leakage (with the instrument's speckle field), a uniform
sky of unit radiance and the scene's disk are rendered once, without planets,
by the same program as `photon_rates`. Each image is then summed in the
aperture at every position of a half-pixel lattice, by one convolution. The
planet table is the response throughput times the star's photon flux
density, the collecting area and the optical throughput, per unit
planet-to-star contrast. The collecting area, optical throughput and the
instrument's `flux_scale` come from one product, `instrument.effective_area_m2`,
so a record carries the scale of the instrument it was built for. The record
refuses a response built for another instrument: another coronagraph, detector
pixel scale, band set or set of wavelength nodes raises a `ValueError`, so a
coronagraph swapped with `equinox.tree_at` needs a new response.

**Read.** A planet is an east and a north offset from the star and a contrast, observed
at a time. The planet rate is read linearly between rings and scaled by the
contrast. The backgrounds are read bilinearly at the detector position the
planet falls on, and the uniform-sky read is scaled by the local-zodi radiance
of that time.
The read is a pure function of arrays, so it runs under `jax.jit`, `jax.vmap`
and `jax.grad`.

The read returns `optixstuff.Rates` in photons/s/nm at the exposure's
wavelength nodes, with the aperture's pixel count in `n_pix` and a second
record in `variance` summed with squared weights. The detector's yield and
noise enter afterwards, so one record serves any detector setting, frame time
and post-processing assumption.

## Local zodi and the observation time

The local zodi on the detector is the instrument's response to a uniform sky
times one number per wavelength: the sky's radiance along the line of sight.
The response belongs to the instrument. The radiance depends on where the
target sits relative to the Sun, which changes through the year: a target on
the ecliptic is several times brighter in zodi 45 degrees from the Sun than
opposite it.

The record therefore holds the two parts separately. It tabulates the response
per unit radiance (1 photon/s/m^2/nm/sr) and keeps what the radiance is
computed from: `scene.local_zodi` and the scene's sky frame (right ascension,
declination and epoch, `scene.sky_frame`), held as `record.local_zodi` and
`record.sky_frame` as they were at build time. A scene with no sky position is
allowed only with a local zodi that ignores the sightline, such as `AYOZodi`; a
pointing-dependent model raises a `ValueError` when it is read. Each read takes
`time_d`, days from that epoch, and

1. evaluates `record.local_zodi.photon_radiance(nodes_nm, sky_frame, time_d)` at
   the record's wavelength nodes; the model derives whatever it needs from the
   frame and the time (`LeinertZodi` finds the star's ecliptic latitude and its
   longitude from the Sun with `orbix.observatory.ecliptic_pointing`),
2. multiplies the uniform-sky read by it.

So one record per target, instrument and band set serves every observation
time; nothing is rebuilt as the year goes by. `time_d` is a required keyword
with no default: the time a record was built at says nothing about when it is
read, and a default would return the zodi of the wrong season without a sign.

The radiance costs a table lookup per wavelength node. Under `jax.vmap` a
`time_d` shared by a batch of planets (closed over, or `in_axes=None`) is
evaluated once for the batch; a `time_d` batched alongside the planets is
evaluated per read. Both stay inside the read budget below.

Only the local zodi follows `time_d`. The star's flux, the disk and any
speckle realization are those of the exposure the record was built from.

## Roll

Star leakage, local zodi and the pixel counts are fixed on the detector. The
exozodi is fixed on the sky. `aperture_rates(..., roll_deg=...)` reads the
first group where the planet falls at the requested roll and the exozodi where
it fell at the roll the record was rendered at. `roll_deg` is the position angle of detector +y, as in
`build_exposure`.

## S/N and exposure time

The reduction belongs to optixstuff, which the jaxedith analytic path also
ends at; [how an exposure time is computed](https://optixstuff.readthedocs.io/en/latest/explanation/exposure_time.html) follows both roads. In short:

`snr(rates, detector, time_s, frame_time_s=..., post_processing=...)` and
`exposure_time(rates, detector, target_snr, frame_time_s=..., post_processing=...)`
take the read.
With planet electrons/s `p`, the detector variance of one frame divided by
its live seconds `b`, and the squared residual rate `f2`:

- `S/N = p t / sqrt(b t + f2 t**2)`
- `t = snr**2 b / (p**2 - snr**2 f2)`

`b` comes from `detector.mean_and_variance`: source shot noise from the
squared-weight rates, dark current, clock-induced charge and read noise over
the aperture's pixels. `optixstuff.PostProcessing` sets the residual speckle and exozodi
fractions behind `f2` and a `background_variance_factor` that multiplies the
variance of every source but the planet. The factor is a required argument:
two for differential imaging, as in AYO and pyEDITH, one when no reference
image is subtracted. Both calculators require the `post_processing` argument:
there is no default, so no residual and a factor of one must be stated as
`PostProcessing(background_variance_factor=1.0)`.

`optixstuff.exposure_time` returns `optixstuff.ExposureTime(time_s, valid)`. The time is continuous;
round it to whole frames with `optixstuff.frames_for_time`. A target that no
time can reach, because the planet is at or below the residual floor, has
`valid` False and a time of zero, so no infinity enters later arithmetic.

```python notest
import jax
import optixstuff as ox
import coronagraphoto as cp

post_processing = ox.PostProcessing(1 / 30, 1 / 30, background_variance_factor=2.0)
exposure = cp.build_exposure(scene, instrument, bands, time_d=0.0)
response = cp.aperture_response(instrument, exposure.quadrature)
record = cp.aperture_record(scene, instrument, exposure, response)


def time_to_snr(east_arcsec, north_arcsec, contrast, time_d):
    rates = cp.aperture_rates(
        record, east_arcsec, north_arcsec, contrast, time_d=time_d
    )
    return ox.exposure_time(
        rates,
        instrument.detector,
        7.0,
        frame_time_s=100.0,
        post_processing=post_processing,
    )


# One observation time for the whole batch of planets, then a batch of times.
per_planet = jax.vmap(time_to_snr, in_axes=(0, 0, 0, None))
times = jax.jit(per_planet)(easts, norths, contrasts, 120.0)
by_time = jax.jit(jax.vmap(per_planet, in_axes=(None, None, None, 0)))
times = by_time(easts, norths, contrasts, observation_times_d)
```

`scene_aperture_rates(record, scene, time_d)` is that read mapped over every
planet of a scene and every time in `time_d`, with the positions and contrasts
taken from the scene at the record's wavelength nodes. Its `Rates` fields have
shape `(W, Q, K, T)`, so `optixstuff.exposure_time(rates, detector, 7.0,
frame_time_s=100.0, post_processing=...)` returns one time per band, planet and time; a planet with
`scene.valid` False reads zero and its time is invalid.

## What the tables do not hold

- The aperture is a circle centred on the planet, fixed in angle within a
  band: its radius is `rule.radius_lod` at the band centre. Other aperture
  rules are not tabulated.
- The planet throughput is a ring mean. It hides the dependence on azimuth
  and on where the planet falls within a pixel;
  `ApertureResponse.throughput_ring_std` reports how much.
- Rates are NaN where the aperture leaves the detector, where the separation
  leaves the planet table, and where the coronagraph has no response model.
  S/N is then NaN and `valid` is False.
- Apart from the local zodi, a record holds one time: the star's flux, the
  disk and any speckle realization are those of `exposure.time_d`.
- The star's sky position is the one the scene held when the record was
  built; a scene edited afterwards needs a new record.
- Thermal and companion terms are zero, as in the image readers.

## Agreement with the images

An aperture sum of `expected` images and a read of the record use the same
coronagraph products and the same pixel weights. On the validation scene (a
Gaussian YIP, a disk, a local zodi, two bands) the read reproduces the image
sums to rounding for the uniform leakage and zodi, to about 1e-4 for the
exozodi, and to 2e-3 for the planet on 0.2 lambda/D pixels, falling to 2e-4
on 0.1 lambda/D pixels. A read at a time other than the build time reproduces
the aperture sum of local-zodi images rendered at that time to rounding
(4e-15 relative). A read with its S/N and exposure time costs about 0.25
microseconds per planet when batched on one CPU core, whether the time is
shared by the batch or differs on every read.
