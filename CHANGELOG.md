# Changelog

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
