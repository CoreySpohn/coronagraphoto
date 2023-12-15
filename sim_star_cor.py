from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
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
time = Time(2000, format="decimalyear")
wavelength = 500 * u.nm
frac_bandwidth = 0.15
bandpass = SpectralElement(
    Gaussian1D,
    mean=wavelength,
    stddev=frac_bandwidth * wavelength / np.sqrt(2 * np.pi),
)

obs_scen = {
    "diameter": 8 * u.m,
    "wavelength": wavelength,
    "time": time,
    "exposure_time": 2 * u.hr,
    "frame_time": 1 * u.hr,
    "include_star": True,
    "include_planets": True,
    "include_disk": False,
    "bandpass": bandpass,
    "spectral_resolution": 100,
    "do_snr_check": True,
    "return_spectrum": True,
    "wavelength_resolved_flux": True,
    "wavelength_resolved_transmission": True
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
    obs.snr_check(np.arange(1, 100, 1) * u.hr)

    breakpoint()
    plt.imshow(obs.image, norm=colors.LogNorm())
    # re.render(system, coro, observing_scenario, obs)
    plt.close("all")
