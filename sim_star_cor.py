import datetime
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
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

high_fid = {
    "diameter": 8 * u.m,
    "wavelength": wavelength,
    "time": time,
    "exposure_time": 48 * u.hr,
    "frame_time": 1 * u.hr,
    "include_star": True,
    "include_planets": True,
    "include_disk": True,
    "bandpass": bandpass,
    "spectral_resolution": 100,
    "return_spectrum": True,
    "return_frames": True,
    "return_sources": True,
    "wavelength_resolved_flux": True,
    "wavelength_resolved_transmission": True,
}
med_fid = {
    "diameter": 8 * u.m,
    "wavelength": wavelength,
    "time": time,
    "exposure_time": 48 * u.hr,
    "frame_time": 1 * u.hr,
    "include_star": True,
    "include_planets": True,
    "include_disk": True,
    "bandpass": bandpass,
    "spectral_resolution": 100,
    "return_spectrum": False,
    "return_frames": False,
    "return_sources": False,
    "wavelength_resolved_flux": True,
    "wavelength_resolved_transmission": True,
}
low_fid = {
    "diameter": 8 * u.m,
    "wavelength": wavelength,
    "time": time,
    "exposure_time": 48 * u.hr,
    "include_star": True,
    "include_planets": True,
    "include_disk": True,
    "bandpass": bandpass,
    "return_spectrum": False,
    "return_frames": False,
    "return_sources": False,
    "wavelength_resolved_flux": False,
    "wavelength_resolved_transmission": False,
}
high_fid_scen = observing_scenario.ObservingScenario(high_fid)
med_fid_scen = observing_scenario.ObservingScenario(med_fid)
low_fid_scen = observing_scenario.ObservingScenario(low_fid)

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
    obs = observation.Observation(coro, system, high_fid_scen)
    # obs = observation.Observation(coro, system, med_fid_scen)
    obs.create_count_rates()

    start = datetime.datetime.now()
    obs.create_count_rates()
    obs.count_photons()
    print(f"High fidelity: {datetime.datetime.now() - start}")

    obs.load_observing_scenario(med_fid_scen)
    start = datetime.datetime.now()
    obs.create_count_rates()
    obs.count_photons()
    print(f"Medium fidelity: {datetime.datetime.now() - start}")

    obs.load_observing_scenario(low_fid_scen)
    start = datetime.datetime.now()
    obs.create_count_rates()
    obs.count_photons()
    print(f"Low fidelity: {datetime.datetime.now() - start}")

    plt.imshow(obs.data["total"], norm=colors.LogNorm(), origin="lower")
    x_loc = obs.system.planets[4]._x_pix[0] - obs.system.star._x_pix[0]
    y_loc = obs.system.planets[4]._y_pix[0] - obs.system.star._y_pix[0]
    plt.scatter(y_loc.value, x_loc.value, marker="x", color="r")
    plt.scatter(120, 120, marker="x", color="g")
    plt.show()

    obs.snr_check(np.arange(1, 100, 1) * u.hr)

    # re.render(system, coro, observing_scenario, obs)
    plt.close("all")
