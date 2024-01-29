import astropy.units as u
from astropy.time import Time
from synphot import SpectralElement


class ObservingScenario:
    def __init__(self, scenario=None):
        default_scenario = {
            "diameter": 1 * u.m,
            "wavelengths": 500 * u.nm,
            "times": Time(2000, format="decimalyear"),
            "exposure_time": 1 * u.d,
            "frame_time": 1 * u.hr,
            "include_star": False,
            "include_planets": True,
            "include_disk": False,
            "include_photon_noise": True,
            "return_spectrum": False,
            "bandpass": None,
            "detector_shape": None,
            "detector_pixel_scale": None,
        }
        self.scenario = default_scenario
        if scenario is not None:
            self.scenario.update(scenario)

    def get_subscenario(self, time, wavelength):
        subscenario = self.copy()
        subscenario["times"] = time
        subscenario["wavelengths"] = wavelength
        return subscenario
