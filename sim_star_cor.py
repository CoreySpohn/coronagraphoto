from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from exoverses.exovista.system import ExovistaSystem
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.models import (BlackBodyNorm1D, Box1D, Empirical1D, Gaussian1D,
                            GaussianFlux1D)

from coronagraphoto import (coronagraph, observation, observations,
                            observing_scenario, render_engine)

# Input files
scene = Path("input/scenes/999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.00.fits")
# Parameters
# bandpass = SpectralElement(
#     Gaussian1D,
#     mean=mode["lam"],
#     stddev=mode["deltaLam"] / np.sqrt(2 * np.pi),
# )
time = Time(2000, format="decimalyear")
wavelength = 500 * u.nm
frac_bandwidth = 0.15
bandpass = SpectralElement(
    Gaussian1D,
    mean=wavelength,
    stddev=frac_bandwidth * wavelength / np.sqrt(2 * np.pi),
)
# boxbandpass = SpectralElement(Box1D, x_0=wavelength, width=frac_bandwidth * wavelength)
# wave = [999, 1000, 2000, 3000, 3001]  # Angstrom
# thru = [0, 0.1, -0.2, 0.3, 0]
# bp = SpectralElement(Empirical1D, points=wave, lookup_table=thru, keep_neg=True)
# bandpass = SpectralElement.from_filter("johnson_v")

# black_body = SourceSpectrum(BlackBodyNorm1D, temperature=5778 * u.K)
# gaussian_absorption = SourceSpectrum(
#     GaussianFlux1D, amplitude=3 * u.mJy, mean=550 * u.nm, stddev=20 * u.nm
# )
# gaussian_emission = SourceSpectrum(
#     GaussianFlux1D,
#     total_flux=3.5e-13 * u.erg / (u.cm**2 * u.s),
#     mean=600 * u.nm,
#     fwhm=20 * u.nm,
# )
# spectrum = black_body - gaussian_absorption + gaussian_emission
# spectrum.plot()
# plt.savefig("combined_spectrum.png")
# plt.close("all")

# obs = Observation(spectrum, bandpass, binset=bandpass.waveset)
# binflux = obs.sample_binned(flux_unit="count", area=1 * u.m**2)
# plt.plot(obs.binset, binflux, drawstyle="steps-mid")
# plt.xlabel("Wavelength (Angstrom)")
# plt.ylabel("Flux (count)")
# plt.title("Observation with Johnson V bandpass and custom spectrum")
# plt.savefig("Observation")

# spectrum.plot()
# plt.savefig("test_spectrum.png")

obs_scen = {
    "diameter": 8 * u.m,
    "wavelength": wavelength,
    "time": time,
    "exposure_time": 10 * 24 * u.hr,
    "frame_time": 1 * u.hr,
    "include_star": True,
    "include_planets": True,
    "include_disk": True,
    "return_spectrum": True,
    "bandpass": bandpass,
    "spectral_resolution": 100,
    # "include_photon_noise": True,
}
observing_scenario = observing_scenario.ObservingScenario(obs_scen)
re = render_engine.RenderEngine()

# Initialize coronagraph object.
# Load ExoVista scene
system = ExovistaSystem(scene)

coronagraph_dir1 = Path("input/coronagraphs/LUVOIR-B-VC6_timeseries/")
coronagraph_dir2 = Path(
    "input/coronagraphs/LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/"
)
cdirs = [coronagraph_dir2, coronagraph_dir1]

# Loop over coronagraphs and simulate observations
for cdir in cdirs:
    coro = coronagraph.Coronagraph(cdir)
    obs = observation.Observation(coro, system, observing_scenario)
    breakpoint()
    # re.render(system, coro, observing_scenario, obs)
    plt.close("all")
