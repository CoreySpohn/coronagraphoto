# Changelog

## [7.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v6.1.0...v7.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **simulation:** read the exposure pointing and refuse pointing offsets until rendered
* **coronagraphoto:** close the render and aperture forwards at their working precision
* **coronagraphoto:** share multiband wavelength dispatch across forward reads and keep ring support identical on host and compiled reads
* **simulation:** a band-averaged coronagraph product is placed at one lambda/D scale per observing band, so its band images and aperture responses change; build_exposure takes allow_wide_bands and refuses wide or straddling bands by default.
* **aperture:** aperture_rates refuses a planet or contrast input that does not broadcast to the record's (W, Q); an extra leading axis was previously read as the mean and variance inputs.
* **simulation:** build_exposure now splits bands at the wavelength knots a disk declares (an ExoVista contrast cube), so an exposure of such a scene has more nodes and a more accurate band integral; disks that declare no breaks are unchanged.
* **aperture:** ApertureRecord.planet_per_contrast is replaced by planet_response (effective area times aperture throughput) and star (the node stellar photon flux density at build); aperture_rates takes exactly one of the keyword-only contrast= or planet=; scene_aperture_rates reads Sources.planets, so a thermal planet around a variable star no longer follows the star's variation
* **simulation:** the render passes realize(wavelength_nm=..., elapsed_s=speckle.elapsed_s(time_d)) to Instrument.speckle, so a speckle field must implement realize with the elapsed_s keyword (seconds since its clock origin) and must not subtract epoch_d itself. Under x32, rendering with a field whose epoch_d is nonzero now raises ValueError instead of returning a result quantized to the float32 spacing of the epoch; enable x64, move the scene epoch to the observation so that epoch_d is zero, or anchor the field at zero.
* **simulation:** build_exposure no longer passes the scene's knot leaves to Exposure.reads, since table knots are static; a scene holding a NumPy leaf is refused with the leaf's path.
* **simulation:** the default field_lod is now the instrument's declared field (the detector half-diagonal), so default node counts rise on wide detectors and results move at the 1e-4 level or less. Pass field_lod=0.0 for the previous layout.
* **simulation:** Exposure now carries a Resolution and the knot leaves it reads, build_exposure takes field_lod, and a hand-built Exposure needs a Resolution for tabulated sources.
* **aperture:** read zero with Rates.supported False instead of NaN beyond the tables, the detector and the stamp
* **simulation:** render only what the coronagraph data hold and flag unsupported pixels in Rates.supported
* **aperture:** render and record the warm telescope's thermal emission
* **aperture:** read MultiBandCoronagraph.coronagraph_at
* **aperture:** state post_processing in every calculator call and document it as required
* **simulation:** read MultiBandCoronagraph.coronagraphs
* **aperture:** refuse a response built for another instrument and take the flux scale from effective_area_m2
* **calculator:** state background_variance_factor in post-processing
* **calculator:** delete snr, exposure_time, ExposureTime and ResidualFractions in favor of optixstuff and drop the yippy runtime dependency
* **aperture:** name the record tables, pass the sky frame to the zodi model and drop the position checks and unit-sky subclass

### Features

* **aperture:** add scene_aperture_rates over a scene's planets and times ([9f2c485](https://github.com/CoreySpohn/coronagraphoto/commit/9f2c485496f225171a00936cba29d85b16c236b1))
* **aperture:** name the record tables, pass the sky frame to the zodi model and drop the position checks and unit-sky subclass ([481a6eb](https://github.com/CoreySpohn/coronagraphoto/commit/481a6eb768a3eeab480d350479a2f6f8096b4832))
* **aperture:** read MultiBandCoronagraph.coronagraph_at ([cf1d8ef](https://github.com/CoreySpohn/coronagraphoto/commit/cf1d8ef8b673cd2fd1d999a96ab9adf6cf4d9f5b))
* **aperture:** refuse a response built for another instrument and take the flux scale from effective_area_m2 ([d18579c](https://github.com/CoreySpohn/coronagraphoto/commit/d18579c3aa485633fdb9b11efb9b83c0f96811cf))
* **aperture:** render and record the warm telescope's thermal emission ([6a06c6e](https://github.com/CoreySpohn/coronagraphoto/commit/6a06c6eda2090ae5ec2b9f890d69074b994b23d4))
* **aperture:** state post_processing in every calculator call and document it as required ([faf6ed7](https://github.com/CoreySpohn/coronagraphoto/commit/faf6ed7fb10628dbf9b170f6a5a3108326fc79b6))
* **simulation:** build_exposure declares no reads and refuses NumPy leaves in the scene ([9de9b29](https://github.com/CoreySpohn/coronagraphoto/commit/9de9b29ae330cf22f2f3d1f6d45119a0e1d8dd50))
* **simulation:** build_exposure takes its default field_lod from the instrument and the routes render real nodes only ([82b4744](https://github.com/CoreySpohn/coronagraphoto/commit/82b4744bfeb3d6625a1a1628c623039ccac17474))
* **simulation:** read scene tables through Scene.sources on a resolved exposure ([f53b7b7](https://github.com/CoreySpohn/coronagraphoto/commit/f53b7b7a9450d187c3ccec33b0f9a57eb7ab12f5))
* **simulation:** read the exposure pointing and refuse pointing offsets until rendered ([c304fb2](https://github.com/CoreySpohn/coronagraphoto/commit/c304fb2cf7cbf9f6c4e31a544530cc293d6639bd))
* **simulation:** records carry static_supported beside supported, and expected and photon_rates keep the static light beyond a speckle field on request ([93acc80](https://github.com/CoreySpohn/coronagraphoto/commit/93acc805b0a31f80924d119e2fce872bf93f925e))
* **simulation:** render band-averaged products at the observing band centre ([fbfd68a](https://github.com/CoreySpohn/coronagraphoto/commit/fbfd68ac23b6538cea014474e2bc3d2720edf7a0))
* **simulation:** speckle support ends at the field's data mask ([3818f79](https://github.com/CoreySpohn/coronagraphoto/commit/3818f79df4a4114f7372ff99543aea7f31b23075))
* **simulation:** split bands at a disk's spectral breaks and add planet_source_support ([1b603bb](https://github.com/CoreySpohn/coronagraphoto/commit/1b603bb8565ff5757f27964b28cc3ec1ee1c72bb))


### Bug Fixes

* **aperture:** image and sum only the real wavelength nodes of each band ([998df50](https://github.com/CoreySpohn/coronagraphoto/commit/998df50974c7b8a0a27be9af966b39b3970467a2))
* **aperture:** probe each band at its middle real node ([d7c8756](https://github.com/CoreySpohn/coronagraphoto/commit/d7c8756624c6130a7b7f6e65dc0e611160a368e5))
* **aperture:** read scene planets by photon flux density ([02d5ecb](https://github.com/CoreySpohn/coronagraphoto/commit/02d5ecbe03895334a97924a46965d5f9efb316f2))
* **aperture:** read zero with Rates.supported False instead of NaN beyond the tables, the detector and the stamp ([8f72447](https://github.com/CoreySpohn/coronagraphoto/commit/8f724471cdace6611e19509aa9a0465d3816b05b))
* **aperture:** refuse a planet or contrast shape that is not (W, Q) and keep float64 precision in the read ([791d9b1](https://github.com/CoreySpohn/coronagraphoto/commit/791d9b12bb177d099689fc8369dce152ffcf94dc))
* **aperture:** run the response guard at compile time so aperture_record works under jit ([150cb58](https://github.com/CoreySpohn/coronagraphoto/commit/150cb5812078dddd8fe4a3158825929dea647e53))
* **coronagraphoto:** close the render and aperture forwards at their working precision ([3279f80](https://github.com/CoreySpohn/coronagraphoto/commit/3279f80bc5b382008a0ca81ffc6393e63978c91b))
* **coronagraphoto:** share multiband wavelength dispatch across forward reads and keep ring support identical on host and compiled reads ([6e8f18f](https://github.com/CoreySpohn/coronagraphoto/commit/6e8f18f310f58949bf419e6eca2e6e6b30d24286))
* **simulation:** build the disk on the coronagraph's scene grid, so sources beyond the stored PSF grid are rendered ([042ec04](https://github.com/CoreySpohn/coronagraphoto/commit/042ec0430f90a1b9b4058128ab83421daea383d6))
* **simulation:** cast scene-derived values to the render owner precision at entry and refuse mixed owners ([6f4fd6a](https://github.com/CoreySpohn/coronagraphoto/commit/6f4fd6ace9547e64f7e6066cce779234a924f89a))
* **simulation:** convert scene time through the speckle field's elapsed_s ([f2f1169](https://github.com/CoreySpohn/coronagraphoto/commit/f2f116902816ae5861f73a5b16bdd0df0a6725eb))
* **simulation:** evaluate disk radiance on the coronagraph source grid ([539000a](https://github.com/CoreySpohn/coronagraphoto/commit/539000a0dc689ec82e85c77163e77048f5c3a5bc))
* **simulation:** feed the disk to the scene operator as cell integrals ([eea42e4](https://github.com/CoreySpohn/coronagraphoto/commit/eea42e42f3c4365f1e4990b65b72f838a31214b8))
* **simulation:** read a sole rendered component in the source-support query and take only breaks from the disk ([22508f9](https://github.com/CoreySpohn/coronagraphoto/commit/22508f9b0ad75cb26a1d2fbaf17ab0724ae43df7))
* **simulation:** read band-averaged planet PSFs at the model's band-scaled wavelength ([9e1c249](https://github.com/CoreySpohn/coronagraphoto/commit/9e1c249627eb1c9d629cd7190958efc4cb28cec6))
* **simulation:** read static_beyond_speckle on the host, so a NumPy or JAX boolean is accepted under filter_jit ([95d44f0](https://github.com/CoreySpohn/coronagraphoto/commit/95d44f06e504a133cc580d2b9e2609a98beeaabd))
* **simulation:** refuse NumPy leaves at every entry through the require_tree walk, and restore the exposure perturbation loop ([a8bd880](https://github.com/CoreySpohn/coronagraphoto/commit/a8bd880463281eca4c169d6664b0fd1b24b4d66a))
* **simulation:** render only what the coronagraph data hold and flag unsupported pixels in Rates.supported ([ebdd8a7](https://github.com/CoreySpohn/coronagraphoto/commit/ebdd8a766edffd01a59a8f74fc1c6b83a7fd78cf))
* **simulation:** select the source-support component among the rendered ones in the node dtype ([2d8d124](https://github.com/CoreySpohn/coronagraphoto/commit/2d8d124bb9fced046be6ddfd9258d3b7bd3b0ee0))
* **simulation:** skip the node checkpoint under disable_jit so eager runs keep concrete values ([26ec8bb](https://github.com/CoreySpohn/coronagraphoto/commit/26ec8bb193b732c51017986207854ef2c0865b6c))
* **simulation:** support of the stellar, speckle and sky products follows the read resample performs, so coarse detector pixels that deposit lose the edge ring (E7 reference 1000 nm: no pixel of the stellar row moves) ([911c632](https://github.com/CoreySpohn/coronagraphoto/commit/911c6326494323600e9fdd9de70d50db38d081b4))


### Performance Improvements

* **simulation:** checkpoint the per-node render so reverse mode recomputes one node instead of storing all ([9975e17](https://github.com/CoreySpohn/coronagraphoto/commit/9975e170273c099ed3e4e081ec11f8f1d37311c7))
* **simulation:** read band-averaged planet PSFs once per band ([a265976](https://github.com/CoreySpohn/coronagraphoto/commit/a265976c1ce96f3b5fc81e8a98098e9785cdbea0))


### Code Refactoring

* **calculator:** delete snr, exposure_time, ExposureTime and ResidualFractions in favor of optixstuff and drop the yippy runtime dependency ([ea2d54d](https://github.com/CoreySpohn/coronagraphoto/commit/ea2d54dffc4fc69c032ea86c39a12d469ef11343))
* **simulation:** read MultiBandCoronagraph.coronagraphs ([4a4b597](https://github.com/CoreySpohn/coronagraphoto/commit/4a4b59747edc5b53cc9d2195ba159d468981d6c6))


### Tests

* **calculator:** state background_variance_factor in post-processing ([73b69e3](https://github.com/CoreySpohn/coronagraphoto/commit/73b69e375be332262de346b46a770e3eb49d8bc3))

## [6.1.0](https://github.com/CoreySpohn/coronagraphoto/compare/v6.0.0...v6.1.0) (2026-11-20)


### Miscellaneous Chores

* release 6.1.0 ([66c35a9](https://github.com/CoreySpohn/coronagraphoto/commit/66c35a9ece6a717ff665ad9df547e1a7cbb2801e))

## [6.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v5.0.0...v6.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* integrate radiance through Image measures and apply the detector response once
* take a quadrature in the image readers and honour declared spectral breaks
* check imaging and scene capabilities on the host and render on named product grids
* read stellar angular diameter through the stellar interface
* remove datasets module; fetch from skyscapes and yippy
* local-zodi models come from zodi.jax
* **simulation:** consume speckle Images on their own grid and keep the excess signed
* render images as the per-pixel reader of the shared photon-rate budget with one orientation and one detector draw

### Features

* accept a shared quadrature, declare image reconstruction, and close the calculator-to-image seam ([ea387eb](https://github.com/CoreySpohn/coronagraphoto/commit/ea387eb00ea93557f80c8f8ddd063fe12f214373))
* check imaging and scene capabilities on the host and render on named product grids ([18bebbe](https://github.com/CoreySpohn/coronagraphoto/commit/18bebbe715d0a7412ecb2ba87aa25fb512ee2f20))
* render images as the per-pixel reader of the shared photon-rate budget with one orientation and one detector draw ([57030d3](https://github.com/CoreySpohn/coronagraphoto/commit/57030d3a511fbced62783c9173ec9e40c419bc12))
* **simulation:** consume speckle Images on their own grid and keep the excess signed ([9ba8a77](https://github.com/CoreySpohn/coronagraphoto/commit/9ba8a77ae3176a2ea86ed6e74b8c85abce59e8d3))
* take a quadrature in the image readers and honour declared spectral breaks ([2473548](https://github.com/CoreySpohn/coronagraphoto/commit/2473548e48597a784366278d28101448298c1048))


### Bug Fixes

* **simulation:** integrate coronagraph products once, directly over detector pixels ([4a6b12d](https://github.com/CoreySpohn/coronagraphoto/commit/4a6b12d8a913577864af9fe52313f5bd95b3ba7c))


### Performance Improvements

* cache compiled reader kernels per static configuration and accept the optical path as a jit argument ([9880dd7](https://github.com/CoreySpohn/coronagraphoto/commit/9880dd7b1bff11df49321b0d816497c3c186029c))


### Code Refactoring

* integrate radiance through Image measures and apply the detector response once ([8d3dcd5](https://github.com/CoreySpohn/coronagraphoto/commit/8d3dcd5403637f62682236b63594f0109bec0626))
* local-zodi models come from zodi.jax ([3db9346](https://github.com/CoreySpohn/coronagraphoto/commit/3db93462bf541108881d18e294271eee82ac167f))
* read stellar angular diameter through the stellar interface ([040535c](https://github.com/CoreySpohn/coronagraphoto/commit/040535cf4059fd2e459e865df102d77dfd07caf2))
* remove datasets module; fetch from skyscapes and yippy ([3dcfd5f](https://github.com/CoreySpohn/coronagraphoto/commit/3dcfd5faa9ccfaadebe38d19fd7e01ce9f55727e))

## [5.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v4.2.0...v5.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **simulation:** one scene ephemeris per exposure; times are days from the scene epoch

### Features

* **simulation:** one scene ephemeris per exposure; times are days from the scene epoch ([d12e549](https://github.com/CoreySpohn/coronagraphoto/commit/d12e54947b5c5fa967f8d2ab68b5f52e71836499))

## [4.2.0](https://github.com/CoreySpohn/coronagraphoto/compare/v4.1.0...v4.2.0) (2026-11-05)


### Bug Fixes

* **simulation:** place sources in the FITS display convention and define telescope_pa_deg as the sky PA of detector +y ([1573619](https://github.com/CoreySpohn/coronagraphoto/commit/15736199fdbdff61a8d4db6951722ca6c0aacc7f))


### Documentation

* **simulation:** state the skyscapes disk pa_deg convention and the per-source placement change ([252fdcb](https://github.com/CoreySpohn/coronagraphoto/commit/252fdcb4a92901eb3ede49cf5306bcfcc08c9370))

### Migration notes

- Detector frames follow the FITS display convention: at
  `telescope_pa_deg = 0`, rows increase north and columns increase west.
  Display with `imshow(origin="lower")` for north up and east left.
- `telescope_pa_deg` is the sky position angle of detector +y, measured
  north through east. A source at sky offset (E, N) lands at detector
  x = -E cos(theta) + N sin(theta), y = N cos(theta) + E sin(theta);
  planets and disks share this mapping. With the earlier orbix and
  skyscapes releases, planets built from orbital elements arrived as
  (north, east) and were placed a quarter-turn away from this;
  ExoVista planets and cubes already arrived as (west, north) and keep
  their zero-roll placement.
- Require `orbix>=3.0.0` and `skyscapes>=2.1.0` for consistent sky coordinates
  and FITS-oriented disk images.

## [4.1.0](https://github.com/CoreySpohn/coronagraphoto/compare/v4.0.0...v4.1.0) (2026-10-13)


### Features

* **simulation:** return per-source rate maps from system_rate under trace ([c937b18](https://github.com/CoreySpohn/coronagraphoto/commit/c937b1815dccd8c53db5bbed80a0f49d936a19e7))

## [4.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v3.1.0...v4.0.0) (2026-07-20)


### ⚠ BREAKING CHANGES

* **simulation:** rates render at detector sampling via the sampling-explicit coronagraph contract -- the detector grid is the only grid; speckle plane decoupled from the coronagraph plate scale

### Features

* **loaders:** pass load_disk through to skyscapes; rewrite README quick start for the rate/readout API ([7ce6676](https://github.com/CoreySpohn/coronagraphoto/commit/7ce6676a1cf9b4a51ebdb80fa0d77c8d28825305))
* **simulation:** rates render at detector sampling via the sampling-explicit coronagraph contract -- the detector grid is the only grid; speckle plane decoupled from the coronagraph plate scale ([41d6f73](https://github.com/CoreySpohn/coronagraphoto/commit/41d6f7324ae500cbdec4ffab92cc505a0552845d))
* **speckle:** add speckle_rate/readout and system wiring ([1e5bfcb](https://github.com/CoreySpohn/coronagraphoto/commit/1e5bfcb7c6fd30cf7e2286238996bd64cd08f983))
* **speckle:** speckle_rate asserts the speckle/coronagraph plate-scale equality (was docstring-only) -- adversarial-review hardening ([14462a4](https://github.com/CoreySpohn/coronagraphoto/commit/14462a4593222b1c4c9e5d7614277b703b8b5aac))

## [3.1.0](https://github.com/CoreySpohn/coronagraphoto/compare/v3.0.0...v3.1.0) (2026-06-23)


### Features

* Add citation file and badge ([12b7734](https://github.com/CoreySpohn/coronagraphoto/commit/12b7734e3ba2616ae83e561051059c938beaa8a5))

## [3.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v2.3.1...v3.0.0) (2026-05-25)


### ⚠ BREAKING CHANGES

* Update docs and match the optixstuff api changes
* SimpleDetector re-export -> IdealDetector (mirrors optixstuff v1.0); ConstantThroughputElement -> ConstantThroughput; ZodiSource -> Zodi and ZodiSource*/SpectrumStar/SimpleStar consumer renames track skyscapes v1.0. Public coronagraphoto API surface is unchanged in name; only the re-exported symbols from sibling libraries follow their renames.
* gen_*_count_rate renamed to *_rate; sim_* renamed to *_readout; coronagraphoto.core and coronagraphoto.core.simulation paths removed (use top-level coronagraphoto or coronagraphoto.simulation); new system_rate added as differentiable counterpart to system_readout

### Features

* flatten subpackages and unify simulation API as *_rate/*_readout ([494c266](https://github.com/CoreySpohn/coronagraphoto/commit/494c266fc5f62ab61775f200526fc6282945512f))
* follow skyscapes/optixstuff renames in re-exports and call sites ([ed143c7](https://github.com/CoreySpohn/coronagraphoto/commit/ed143c7a21ead23b4383fdca8ab3f26dcfd729fb))
* Update docs and match the optixstuff api changes ([d12ed23](https://github.com/CoreySpohn/coronagraphoto/commit/d12ed2311725229e2211329b4e06342d30967ed9))


### Bug Fixes

* Add mermaid rendering to docs ([98bcb7e](https://github.com/CoreySpohn/coronagraphoto/commit/98bcb7e9973ce09e0fa12b5c75c5c734529f55a4))

## [2.3.1](https://github.com/CoreySpohn/coronagraphoto/compare/v2.3.0...v2.3.1) (2026-05-24)


### Bug Fixes

* **coronagraphoto:** Pass scalar wavelength to atmosphere in gen_planet_count_rate ([6edcb5f](https://github.com/CoreySpohn/coronagraphoto/commit/6edcb5f12601b392469819e39db57e4d3fb9423b))

## [2.3.0](https://github.com/CoreySpohn/coronagraphoto/compare/v2.2.2...v2.3.0) (2026-05-21)


### Features

* Add support for new skyscapes disks ([81c24dd](https://github.com/CoreySpohn/coronagraphoto/commit/81c24dd3889cbf663839dad52ee3a5232d58db90))
* Update to match skyscapes API for loading exovista data ([5cdaf5f](https://github.com/CoreySpohn/coronagraphoto/commit/5cdaf5f91db136485fceac78ecdf0c58eec0971d))

## [2.2.2](https://github.com/CoreySpohn/coronagraphoto/compare/v2.2.1...v2.2.2) (2026-05-14)


### Bug Fixes

* Updating zenodo citation details ([9b9b8bb](https://github.com/CoreySpohn/coronagraphoto/commit/9b9b8bbc0a67b6d3a9c5a4df7910ae2c8616abac))

## [2.2.1](https://github.com/CoreySpohn/coronagraphoto/compare/v2.2.0...v2.2.1) (2026-02-27)


### Bug Fixes

* Add exoverses dependency ([2ab5b1d](https://github.com/CoreySpohn/coronagraphoto/commit/2ab5b1d7503e0216d74fe592c0f9fb7b507236e4))
* Updating the pooch management ([8810b35](https://github.com/CoreySpohn/coronagraphoto/commit/8810b358085c98bdd68bec08f0a60db76d0d72f5))

## [2.2.0](https://github.com/CoreySpohn/coronagraphoto/compare/v2.1.4...v2.2.0) (2026-01-23)


### Features

* Introduce RGB composite exposure and rendering utilities, remove `name` attributes from source classes, add `alpha_dMag` to `PlanetSource`, and enhance ExoVista loading with planet unit corrections and Earth-like planet identification. ([4060fec](https://github.com/CoreySpohn/coronagraphoto/commit/4060fecb3231c4b6da44e5b2926468dffe45d026))

## [2.1.4](https://github.com/CoreySpohn/coronagraphoto/compare/v2.1.3...v2.1.4) (2025-12-10)


### Bug Fixes

* Make planet loading logic to truncate excess planets and issue a warning ([256100f](https://github.com/CoreySpohn/coronagraphoto/commit/256100fd728ec069a87629056d810f122af77bd5))

## [2.1.3](https://github.com/CoreySpohn/coronagraphoto/compare/v2.1.2...v2.1.3) (2025-12-10)


### Bug Fixes

* Hopefully improve psf_datacube initialization to avoid unnecessary copies and ensure correct dtype handling on GPU ([d4ed75e](https://github.com/CoreySpohn/coronagraphoto/commit/d4ed75eb558e2e2878705f3d51a47fe538568bb2))

## [2.1.2](https://github.com/CoreySpohn/coronagraphoto/compare/v2.1.1...v2.1.2) (2025-12-10)


### Bug Fixes

* Update psf_datacube type to jnp.ndarray and ensure proper initialization ([fb62a43](https://github.com/CoreySpohn/coronagraphoto/commit/fb62a437c3652a748684cef67f644f4f20879fa7))

## [2.1.1](https://github.com/CoreySpohn/coronagraphoto/compare/v2.1.0...v2.1.1) (2025-12-09)


### Bug Fixes

* Add yippy dependency ([48e88fe](https://github.com/CoreySpohn/coronagraphoto/commit/48e88feb08b82fd604de2967bc46c6feedf71db8))

## [2.1.0](https://github.com/CoreySpohn/coronagraphoto/compare/v2.0.0...v2.1.0) (2025-12-08)


### Features

* Implement quarter-symmetric PSF convolution in simulation.py ([f48d5e3](https://github.com/CoreySpohn/coronagraphoto/commit/f48d5e3498ad303ce40c26200f8872f19783aa64))

## [2.0.0](https://github.com/CoreySpohn/coronagraphoto/compare/v1.4.0...v2.0.0) (2025-11-26)


### Miscellaneous Chores

* release 2.0.0 ([303215f](https://github.com/CoreySpohn/coronagraphoto/commit/303215f1c56f03750e16cdcd2cead1d89957aefa))

## [1.4.0](https://github.com/CoreySpohn/coronagraphoto/compare/v1.3.0...v1.4.0) (2025-10-08)


### Features

* Integrate SpeckleMap functionality ([36a3406](https://github.com/CoreySpohn/coronagraphoto/commit/36a34067501e96c3553aad8153d422bef283f5d5))

## [1.3.0](https://github.com/CoreySpohn/coronagraphoto/compare/v1.2.0...v1.3.0) (2025-07-02)


### Features

* Introduce Detector class for modeling detector behavior in coronagraph simulations ([0f6b9f1](https://github.com/CoreySpohn/coronagraphoto/commit/0f6b9f1e6d40217f9d4bd771a2d3f0b6536cd26f))

## [1.2.0](https://github.com/CoreySpohn/coronagraphoto/compare/v1.1.0...v1.2.0) (2025-06-26)


### Features

* Add PostProcessing and ProcessingConfig classes for simple post-processing workflows ([41b5615](https://github.com/CoreySpohn/coronagraphoto/commit/41b5615ec7b59796fbf4f9501ef73e1847635763))
* Implement CompositeObservation class for creating composite images from multiple observations into a single RGB image ([68be0bd](https://github.com/CoreySpohn/coronagraphoto/commit/68be0bd25aa2ec6b421b36171909d76229009804))


### Bug Fixes

* Corrected type casting for full_frames in Observation class ([b525119](https://github.com/CoreySpohn/coronagraphoto/commit/b525119cc3936a19dc3b952749ee9869074e785d))
* Fix frame_time_s and exposure_time_s to not always be default values ([0fd8ceb](https://github.com/CoreySpohn/coronagraphoto/commit/0fd8cebfe64ff9f192c640f8faf6e57e7d5bdf91))

## [1.1.0](https://github.com/CoreySpohn/coronagraphoto/compare/v1.0.1...v1.1.0) (2025-06-25)


### Features

* Add zoom_conserve_flux function for image resampling ([2bbfe15](https://github.com/CoreySpohn/coronagraphoto/commit/2bbfe1535da6bda3262fb9ae6c1cd5e51eb96302))
* Enhance logging with color-coded output ([374045e](https://github.com/CoreySpohn/coronagraphoto/commit/374045e7201dae5e354f4c4aa7c4d7270f9761a3))
* Expand Observation class with detailed simulation capabilities ([fec4e55](https://github.com/CoreySpohn/coronagraphoto/commit/fec4e5586e17701fe051cb34a6b74956b1af48d1))
* Transition to using yippy ([19e7024](https://github.com/CoreySpohn/coronagraphoto/commit/19e702459388051868c60841e9ee8278b87f8ab9))

## [1.0.1](https://github.com/CoreySpohn/coronagraphoto/compare/v1.0.0...v1.0.1) (2024-04-09)


### Bug Fixes

* **main:** Adding zenodo information ([26837e6](https://github.com/CoreySpohn/coronagraphoto/commit/26837e6b894d0380d0dd529506da1cd35d1ccc1d))

## 1.0.0 (2024-03-22)


### Bug Fixes

* adding dependabot ([5a7fe8e](https://github.com/CoreySpohn/coronagraphoto/commit/5a7fe8e613a436155293495a44fb121d98819f16))
* Adding workflow ([4f020b2](https://github.com/CoreySpohn/coronagraphoto/commit/4f020b26ce70578127a3560af6947c1d43b1dfaf))
* Commenting out not implemented code ([49d9b0a](https://github.com/CoreySpohn/coronagraphoto/commit/49d9b0a40beb106a432fe501d4f5d01459162e41))
* fixing more ruff linting things ([1dbb5e9](https://github.com/CoreySpohn/coronagraphoto/commit/1dbb5e90553949a5d9d186df91868a4f48c3ea0a))
* hatch version control ([a1f342c](https://github.com/CoreySpohn/coronagraphoto/commit/a1f342ce8b883076205bb832bbdd9f964bede3fa))
* Linting ([bca15f0](https://github.com/CoreySpohn/coronagraphoto/commit/bca15f0e096ce67466117c92d0cc6487ea11010e))
* removing the problem files for ruff ([41911c1](https://github.com/CoreySpohn/coronagraphoto/commit/41911c1f86c455dd560d054ff65ceeb843c76619))
* Still more annoying linting ([7022c05](https://github.com/CoreySpohn/coronagraphoto/commit/7022c05ec4c1ecf7f0859803a0fc47198a11939c))
