# Thinking through it all
## The structure of an observation
### Raw exposure
There are roughly the following components of a telescope exposure
1. The astrophysical scene
	1. The object we're pointing at. We can approximate the rate at which photons are produced by the components of the scene (e.g. star, disk, planet).
	2. Units - Spectral flux density (photons/second/meter^2/nm)
	3. Tool - ExoVista + coronagraphoto
2. The telescope
	1. Photons hit the primary mirror and are reflected toward the optics. Mirror diameter sets the number of photons available and the telescope's roll angle sets the position angle of the astrophysical scene
	2. Units - Spectral photon flux (photons/second/nm)
3. The optical system
	1. The photons enter the telescope and are transformed by the optical components, in this case the coronagraph
	2. Units - Photons/second
	3. Tool - yippy + coronagraphoto
	4. Eventually this will be adjusted to include an integral field spectrograph which sits behind the coronagraph so we'll have to be able to chain these
4. The detector
	1. Photons hit the detector and are counted by the detector. Noise occurs here based on Poisson statistics (photon noise) and detector noise (read noise, dark current, clock induced charge)
	2. Units - Electrons
	3. Tool - coronagraphoto
Currently ignored: Control loop/AO, actual optics (we have pre-processed coronagraph performance files with PSFs), cosmic rays, more complicated detector effects
### Exposure set
There are a number of reasons we end up taking multiple exposures
- To avoid saturation of the detector and reduce some detector noise values
- To observe a reference star
- To observe the same star at a different roll angle
### Post processing
In post processing we take the raw exposure information and either 
- Compare it to another exposure
	- Angular differential imaging
	- Reference differential imaging
- Apply an algorithm to suppress noise
	- Model based PSF subtraction
- Combine fully processed observations at three different wavelengths into a single RGB image dataset
## The new things
1. Adding speckle noise in the form of the file that Neil gave me
	1. Requires that I can load the speckle field
		1. Neil hands me normalized intensity maps for the stellar intensity information, loop through the off-axis PSFs to get the highest max value in all of them, then multiply the stellar contrast maps by that value to get the YIP stellar_intens.fits 3d datacube.
	2. Assign half of the speckle frames to the science image and half to the reference image
	3. For each exposure choose one of the speckle frames randomly for each
	4. Speckle field replaces the Star information from exovista but needs to be normalized
2. ADI
	1. Need to be able to rotate the SkyScene by rolling the telescope
## Managing the SkyScene
- It would be nice if in the Pipeline you could add the different objects in a modular manner like:
	- AddStar
	- AddPlanets
	- AddDisk
- That would make it easier to define reference star information so as to not include the disk and could make it easier to define separate objects for the star/planet/disk
	- To do the exposure for the reference star we don't need to make the star arbitrarily higher flux, we can just take the expectation value of photons instead of doing Poisson counting (since we normalize both science and reference to the same factor and then subtract them)
## Defining the possible input/output transforms
- Each Transform should know whether it's making a valid operation
	- e.g. ToElectrons should know that it takes in an array of photons and returns an array of electrons whereas a Rotate/Roll transform should know that it works independent of the units
	- That way at run-time we can quickly check the pipeline to ensure it's performing valid operations and raise a detailed error if necessary

# Conclusions
1. "Session" is far too vague because it is attempting to capture too much. We should separate things out into a data collection pipeline and a data processing pipeline.
	1. Need an "Observatory" object that holds all the information on mirror, optical system, detector, roll angle, etc
		1. It's important that we can still get the intermediate data, like count rates and photon counts before detector effects. I think the best way to manage this is by providing something like a "light path" sequence that defines the path of light through the observatory (e.g. \[Primary, Coronagraph, Detector\] (electron counts) vs \[Primary, Coronagraph\] (photon counts)). If an exposure time is None then it remains as a rate of electons/second or photons/second.
	2. Need an "ObservationSequence" which holds a list of PlannedExposure objects that describe what Target is being observed, when, at what wavelength, for how long, etc
		1. We will still want some way to generate an ObservationSequence, it's not something we want to manage manually the majority of the time. Something like a base "ObservationSequence.generate()" method or classes like GenerateSequence, "GenerateRDISequence", or "GenerateADISequence"
        2. The SkyScene should be turned into a Target object which holds the Exovista information (and potentially other things in the future)
	3. We should be able to make a call like `da=Observatory.run(observation_sequence)` which executes observations in the sequence and returns a single xarray Dataset or DataArray
	4. Then we create a ReductionPipeline object of post-processing steps like `pipe = ReductionPipeline(steps=[RDISubtraction(), StackRGB(R=1000 nm, G=700 nm, B = 500 nm)])` and can call `final = pipe.process(da)` to get the final science image
2. It's important that we can take time-varying (or wavelength varying) speckle information and easily slot it into this pipeline.
	1. One potential way to manage this is to have something like a "core loop" during an exposure that iterates through time steps during an exposure and checks if any of the objects need to be updated. So if our control loop is operating once per 10 seconds, which ends up changing the speckle pattern, we do nothing until t=10 and then readout electrons into the exposure consistent with the last count rate map lasting for 10 seconds, change the speckle pattern and wait another 10 seconds. This could be a powerful addition and core feature which provides a lot of flexibility moving forward since almost all components in the chain of astrophysical scene, wavefront error, observatory, optical system, detector can be time varying and wavelength varying.
		1. Actually this could also be the proper way to manage spectral information as well. The object flux values depend on time and wavelength, transmission through the bandpass depends on wavelength, so we need to be able to stack and process all of that information. Not all detectors can read out wavelength/energy information and sometimes this level of fidelity would be a computational burden, but the photons/count rates do need to be processed separately when we add functionality for something like an IFS or energy resolving detectors.
		2. This definitely needs to be better managed than the current system, there has to be some good behavioral pattern to manage things like this. We don't want to be in a loop continually checking for things to happen if we can avoid it.