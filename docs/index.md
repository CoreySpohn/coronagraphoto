# coronagraphoto

Coronagraphic image readers for shared photon and electron budgets.

`photon_rates` returns per-pixel spectral photon integrands; `expected`
streams those nodes through the detector yield into band electron rates.
`observe` draws an electron `Image` on the detector sky grid. Individual source
terms remain accessible through the shared records.

Inputs are a `skyscapes.Scene` (star, planets, disk and local zodi), an
`optixstuff.Instrument`, and an `optixstuff.Exposure` built once with
`build_exposure` from wavelength `Band`s, a scalar time in days from the scene
epoch, and a roll in degrees.

`aperture_record` tabulates one target's aperture rates from the same images,
and `aperture_rates`, `snr` and `exposure_time` read that table once per
planet under `jax.vmap`.

See the [README quick start](https://github.com/CoreySpohn/coronagraphoto#quick-start),
[rate contracts](explanation/rate_vs_readout),
[aperture rates and exposure time](explanation/exposure_time), and
[JAX and memory](explanation/performance).

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
