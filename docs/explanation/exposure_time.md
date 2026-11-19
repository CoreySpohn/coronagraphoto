# Aperture rates, S/N and exposure time

The image readers answer "what does the detector see". A yield or scheduling
code asks a narrower question many millions of times: how long until this
planet reaches a signal-to-noise ratio. Rendering an image per question is far
too slow, so the answer is read from tables that the image path fills once.

## Three steps, three cadences

| Step | Function | When | Cost here |
|---|---|---|---|
| Calibration | `aperture_calibration(instrument, quadrature, rule=...)` | once per instrument and band set | seconds to minutes |
| Record | `aperture_record(scene, instrument, exposure, calibration)` | once per target and exposure | one background render |
| Read | `aperture_rates(record, separation_arcsec, contrast, position_angle_deg=...)` | once per planet | a few gathers |

**Calibration.** Point sources are placed on rings around the star, imaged on
the detector pixels by the coronagraph, and summed in a circular aperture
centred on each source. The ring mean is the planet throughput at that
separation. The calibration also holds the aperture's pixel weights for a
centre at each half-pixel phase. Nothing in it depends on a target.

**Record.** The star leakage (with the instrument's speckle field), the local
zodi of `scene.zodi` and the scene's disk are rendered once, without planets,
by the same program as `photon_rates`. Each image is then summed in the
aperture at every position of a half-pixel lattice, by one convolution. The
planet table is the calibration throughput times the star's photon flux
density, the collecting area and the optical throughput, per unit
planet-to-star contrast.

**Read.** A planet is a separation, a position angle and a contrast. The
planet rate is read linearly between rings and scaled by the contrast. The
backgrounds are read bilinearly at the detector position the planet falls on.
The read is a pure function of arrays, so it runs under `jax.jit`, `jax.vmap`
and `jax.grad`.

The read returns `optixstuff.Rates` in photons/s/nm at the exposure's
wavelength nodes, with the aperture's pixel count in `n_pix` and a second
record in `variance` summed with squared weights. The detector's yield and
noise enter afterwards, so one record serves any detector setting, frame time
and post-processing assumption.

## Roll

Star leakage, local zodi and the pixel counts are fixed on the detector. The
exozodi is fixed on the sky. `aperture_rates(..., roll_deg=...)` reads the
first group where the planet falls at the requested roll and the exozodi where
it fell at the roll the record was rendered at. Position angle is measured
north through east, and `roll_deg` is the position angle of detector +y, as in
`build_exposure`.

## S/N and exposure time

`snr(rates, detector, time_s, frame_time_s=...)` and
`exposure_time(target_snr, rates, detector, frame_time_s=...)` take the read.
With planet electrons/s `p`, the detector variance of one frame divided by
its live seconds `b`, and the squared residual rate `f2`:

- `S/N = p t / sqrt(b t + f2 t**2)`
- `t = snr**2 b / (p**2 - snr**2 f2)`

`b` comes from `detector.mean_and_variance`: source shot noise from the
squared-weight rates, dark current, clock-induced charge and read noise over
the aperture's pixels. `PostProcessing` sets the residual speckle and exozodi
fractions behind `f2`; without it the residual is zero.

`exposure_time` returns `ExposureTime(time_s, valid)`. The time is continuous;
round it to whole frames with `optixstuff.frames_for_time`. A target that no
time can reach, because the planet is at or below the residual floor, has
`valid` False and a time of zero, so no infinity enters later arithmetic.

```python notest
import jax
import coronagraphoto as cp

exposure = cp.build_exposure(scene, instrument, bands, time_d=0.0)
calibration = cp.aperture_calibration(instrument, exposure.quadrature)
record = cp.aperture_record(scene, instrument, exposure, calibration)


def time_to_snr(separation_arcsec, position_angle_deg, contrast):
    rates = cp.aperture_rates(
        record, separation_arcsec, contrast, position_angle_deg=position_angle_deg
    )
    return cp.exposure_time(7.0, rates, instrument.detector, frame_time_s=100.0)


times = jax.jit(jax.vmap(time_to_snr))(separations, angles, contrasts)
```

## What the tables do not hold

- The aperture is a circle centred on the planet, fixed in angle within a
  band: its radius is `rule.radius_lod` at the band centre. Other aperture
  rules are not tabulated.
- The planet throughput is a ring mean. It hides the dependence on azimuth
  and on where the planet falls within a pixel;
  `ApertureCalibration.throughput_ring_std` reports how much.
- Rates are NaN where the aperture leaves the detector, where the separation
  leaves the planet table, and where the coronagraph has no response model.
  S/N is then NaN and `valid` is False.
- A record holds one epoch: the star's flux, the disk and any speckle
  realization are those of `exposure.time_d`.
- Thermal and companion terms are zero, as in the image readers.

## Agreement with the images

An aperture sum of `expected` images and a read of the record use the same
coronagraph products and the same pixel weights. On the validation scene (a
Gaussian YIP, a disk, a local zodi, two bands) the read reproduces the image
sums to rounding for the uniform leakage and zodi, to about 1e-4 for the
exozodi, and to 2e-3 for the planet on 0.2 lambda/D pixels, falling to 2e-4
on 0.1 lambda/D pixels. A read with its S/N and exposure time costs about
0.2 microseconds per planet when batched on one CPU core.
