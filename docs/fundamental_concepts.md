# The Core Physics
Coronagraphoto works by taking an astrophysical flux density computed by ExoVista and propagating it a telescope.
Below is an explanation of the process, the quantities involved, and the units at each step.
## At a glance
The basic propagation is 
$$C(\lambda,t)=F(\lambda,t) \, A \, T_c(\lambda) \, \Delta\lambda \, T_f(\lambda) \, QE(\lambda)$$
1. $F(\lambda, t)$ - Astrophysical Source - Creates a spectral flux density for the source (photons/s/nm/m^2)
	1. Some sources may be considered time-independent (disk/star do not move much)
	2. The disk provides a *grid* of spectral flux density values since it's an extended source
2. $A$ - Primary mirror - Multiplies the count rate by the illuminated area which is a constant (photons/s/nm)
3. $T_\text{c}(\lambda)$ - Coronagraph - Multiplies the count rates by a grid of throughputs that's a function of wavelength since coronagraphs are defined in $\lambda/D$ pixel scale units (photons/s/nm)
4. $T_f(\lambda)$ and $\Delta\lambda$ -  Filter - Multiplies by the throughput of the filter at the wavelength and the bandwidth for the spectral resolution (photons/s)
5. $QE(\lambda)$ - Detector - Counts photons with Poisson statistics based on observation time, converts the incident photons to electrons based on the quantum efficiency (either a float or function of wavelength), and potentially adding noise terms like dark current, read noise, and clock induced count rates (electrons)
	1. This could also remain in electrons/second by multiplying just by the quantum efficiency
Important Note: The planet, star, and disk all require separate propagation methods through the coronagraph layer
### Astrophysical Source
Coronagraphoto uses ExoVista to compute the astrophysical fluxes for the star, disk and planets. To be precise, the fluxes are "spectral flux density" values which have units of $\text{photons}/(\text{s}\,\text{nm}\,\text{m}^2)$. ExoVista disk fluxes are computed on a grid set by the input parameters `npix` (the number of pixels along one axis of the grid) and `pixscale` (the apparent angular separation per pixel in units of arcseconds). These values are functions of wavelength and time in ExoVista. We will represent the flux values with $F_\text{star}$, $F_\text{disk}$, and $F_\text{planets}$.
### Primary mirror
We currently work on the assumption that mirror shape effects are included in the Coronagraph object, thus the primary is simply used to keep track of the illuminated area of the mirror.
### Coronagraph
We define a coronagraph using a "yield input package" which is a standard set of fits files defined by Chris Stark and John Krist. They are generally broken up as follows:
- Stellar Intensity
	- A unitless 3d array of the stellar intensity function $I$, as a function of $(x, y)$ pixel coordinates and the stellar angular diameter $\theta_\textrm{star}$. Values in the map are equal to the stellar count rate in a given pixel divided by the total stellar count rate entering the coronagraph.
- Off-axis PSFs
	- 3d array of PSF maps as a function of $(x, y, k)$. In $(x, y)$ we have standard 2d images that show the PSF. The value at a pixel is the count rate in that pixel divided by the total count rate of the image. The offset $k$ (an integer corresponding to an $(x, y)$ value in the `offax_psf_offset_list.fits` file) represents the astrophysical offset of the source in units of $\lambda / D$. The images are normalized to the total count rate entering the coronagraph.
#### Applying the coronagraph
##### Star
We use the stellar intensity array as a throughput term, multiplying the stellar flux density by the intensity array to get the stellar signal through the coronagraph.
##### Planets
Planets are off-axis point sources, and thus we multiply the stellar flux density by the off-axis PSF at the planet's location in (x,y) space. The off-axis PSFs are interpolated by the tool `yippy`.
##### Disk
Disks are an extended source which poses a unique challenge since the astrophysical scene is a stellar flux density array. The tool `yippy` computes a large datacube with shape $(n_x, n_y, n_x, n_y)$ which has the off-axis PSF for each pixel of the image. This datacube is multiplied by the stellar flux density array with tensor contraction so each spectral flux density pixel is multiplied by the proper PSF for that pixel's location.
### Filter
The filter attenuates unwanted wavelengths. Filters have an intensity function that describes the throughput at a given wavelength. 
- If the observation is in broadband (photometry) the transmission is evaluated at the central wavelength and the bandpass is the equivalent width, which is calculated as $\Delta\lambda_\text{eq} = \int T(\lambda) \text{d}\lambda$
- If a spectral resolution is required, the throughput function is broken up into smaller chunks each of which has its own bandwidth and throughput value. We will eventually want to be able to redo the full propagation for every wavelength of the spectral resolution, but we're going to initially just focus on the photometry.
### Detector
At the detector plane we project the count rates to the pixels on the detector and count the photons and translate them into electrons. The photon counting follows Poisson statistics and the translation to electrons is a Bernoulli process.