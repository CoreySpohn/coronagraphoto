# coronagraphoto

Coronagraphic image readers for shared photon and electron budgets.

`photon_rates` returns per-pixel spectral photon integrands; `expected`
streams those nodes through the detector yield into band electron rates.
`observe` draws an electron `Image` on the detector sky grid. Individual source
terms remain accessible through the shared records.

Inputs are a `skyscapes.Scene` (star, planets, disk and local zodi), an
`optixstuff.Observatory` (a `Telescope` and its named `Instrument`
configurations), an `optixstuff.Exposure` built with `build_exposure` from
wavelength `Band`s, an epoch, an instrument name, a start time in days from that
epoch and a roll in degrees, and an `optixstuff.SpectralSampling` built with
`build_sampling`. `observe` returns an `optixstuff.ExposureData`.

`aperture_record` tabulates one target's aperture rates from the same images,
and `aperture_rates` reads that table once per planet under `jax.vmap`, with
`planet=` for photon flux density or `contrast=` for a build-time stellar hypothesis;
`optixstuff.snr` and `optixstuff.exposure_time` turn the rates into S/N and time.

See the [README quick start](https://github.com/CoreySpohn/coronagraphoto#quick-start),
[rate contracts](explanation/rate_vs_readout),
[aperture rates and exposure time](explanation/exposure_time),
[how an exposure time is computed](https://optixstuff.readthedocs.io/en/latest/explanation/exposure_time.html), and
[JAX and memory](explanation/performance).

## Migrating from 7.x

Three objects replace the one `Instrument` and the one `Exposure` of the 7.x
calls. The `Observatory` holds a `Telescope` (the pupil and the shared
throughput) and the named `Instrument` configurations, so the `primary=` field
is gone and a YIP's pupil comes from `Coronagraph.pupil`. The `Exposure` is
only the commanded event and holds no scene. The `SpectralSampling` holds the
wavelength quadrature and the `Resolution` that `build_exposure` used to attach.
Every reader takes the sampling as a keyword.

`time_d` now counts days from `exposure.epoch`, not from the scene epoch: pass
`epoch=scene.epoch` to keep the old meaning. `observe` returns an
`ExposureData` and `observe_data` is gone. Read the image as `data.image`; the
roll is `data.exposure.pointing.roll_deg`, and the image grid no longer carries
it. A simulated draw has `data.variance` equal to None, and `data.valid` marks
every pixel valid. A sampling or observation built by an older stack is refused
with a `StampMismatchError`.

Before (historical, not executed):

```text
instrument = Instrument(
    primary=coronagraph.primary,
    attenuating_elements=(ConstantThroughput(throughput=0.4),),
    coronagraph=coronagraph,
    detector=detector,
)
exposure = build_exposure(
    scene, instrument, bands, time_d=0.0, roll_deg=30.0, plan=plan
)
rates = expected(scene, instrument, exposure)
image = observe(key, scene, instrument, exposure)
```

After:

```python notest
# scene, coronagraph, detector, bands, plan and key come from the caller.
from optixstuff import ConstantThroughput, Instrument, Observatory, Telescope
from coronagraphoto import build_exposure, build_sampling, expected, observe

observatory = Observatory(
    Telescope(coronagraph.pupil),
    (
        Instrument(
            "imager",
            attenuating_elements=(ConstantThroughput(throughput=0.4),),
            coronagraph=coronagraph,
            detector=detector,
        ),
    ),
)
exposure = build_exposure(
    bands,
    epoch=scene.epoch,
    instrument="imager",
    time_d=0.0,
    roll_deg=30.0,
    plan=plan,
)
sampling = build_sampling(observatory, exposure, scene=scene)
rates = expected(scene, observatory, exposure, sampling=sampling)
data = observe(key, scene, observatory, exposure, sampling=sampling)
image = data.image
```

```{toctree}
:maxdepth: 1
:caption: Get started
:hidden:

installation
dependency-floors
```

<!-- TODO(post-1.0): write tutorials/01_first_simulation and re-add to toctree. -->


```{toctree}
:maxdepth: 1
:caption: Explanation
:hidden:

explanation/rate_vs_readout
explanation/exposure_time
explanation/architecture
explanation/performance
```

```{toctree}
:maxdepth: 1
:caption: How-to
:hidden:

simulating_zodi_with_telescope_orbit
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/index
```
