import copy
import time
from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from exoverses.exovista.system import ExovistaSystem
from synphot import SpectralElement
from synphot.models import Box1D, Gaussian1D
from yippy import Coronagraph

from coronagraphoto import (
    Observation,
    ObservingScenario,
    PostProcessing,
    ProcessingConfig,
    Settings,
)

# Input files
# scene = Path("input/scenes/999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.00.fits")
scene = Path("input/scenes/more_pix.fits")
_time = Time(2000, format="decimalyear")
wavelength = 500 * u.nm
frac_bandwidth = 0.15
bandpass = SpectralElement(
    Gaussian1D,
    mean=wavelength,
    stddev=frac_bandwidth * wavelength / np.sqrt(2 * np.pi),
)
bandpass2 = SpectralElement(Box1D, x_0=wavelength, width=1 * u.nm)

base = "input/settings/example.toml"
diameters = [2, 4, 6, 8] * u.m
# coronagraph_dir1 = Path("../yippy/input/coronagraphs/ApodSol_APLC/")
coronagraph_dir1 = Path("../yippy/input/coronagraphs/eac1/eac1_aavc/")

# Create figure for disk analysis
fig_disk, axs_disk = plt.subplots(len(diameters), 4, figsize=(14, 15))

# Get disk data
system = ExovistaSystem(scene)
disk = system.disk
disk_contrast = disk.contrast[0]  # Get contrast at first wavelength
disk_flux = disk.spec_flux_density(wavelength, _time).squeeze()

# Initialize coronagraph
coro = Coronagraph(coronagraph_dir1, use_jax=True, cpu_cores=12)

# Loop over diameters
for i, diameter in enumerate(diameters):
    # Create observing scenario for this diameter
    med_fid = {
        "diameter": diameter,
        "central_wavelength": wavelength,
        "start_time": _time,
        "exposure_time": 48 * u.hr,
        "frame_time": 48 * u.hr,
        "bandpass": bandpass,
        "spectral_resolution": 100,
    }

    settings = Settings(base)
    settings.include_disk = True
    settings.include_star = True
    settings.include_planets = True
    settings.wavelength_resolved_flux = False
    settings.wavelength_resolved_transmission = False
    settings.return_sources = True
    observing_scenario = ObservingScenario(base, custom_scenario=med_fid)

    # Create observation
    obs = Observation(coro, system, observing_scenario, settings)
    obs.create_count_rates()
    raw_ds = obs.count_photons()

    # Post-process the observation
    processing_config = ProcessingConfig(
        custom_config={
            "star_post_processing_factor": 30,
            "disk_post_processing_factor": 10,
        }
    )
    post = PostProcessing(processing_config)
    processed_ds = post.process(raw_ds)

    # Plot disk contrast (same for all diameters)
    im0 = axs_disk[i, 0].imshow(disk_contrast, norm=colors.LogNorm(), origin="lower")
    axs_disk[i, 0].set_title(f"Disk Contrast (D={diameter.value}m)")
    # if i == len(diameters) - 1:  # Only add colorbar to last row
    #     plt.colorbar(im0, ax=axs_disk[i, 0], label="Contrast")

    # Plot disk spectral flux density (same for all diameters)
    im1 = axs_disk[i, 1].imshow(disk_flux.value, norm=colors.LogNorm(), origin="lower")
    axs_disk[i, 1].set_title(f"Disk Flux Density at {wavelength} (D={diameter.value}m)")
    # if i == len(diameters) - 1:  # Only add colorbar to last row
    #     plt.colorbar(im1, ax=axs_disk[i, 1], label="Flux Density (Jy)")

    # Plot disk count rate (varies with diameter)
    im2 = axs_disk[i, 2].imshow(
        obs.disk_count_rate.value[0, 0], norm=colors.LogNorm(), origin="lower"
    )
    axs_disk[i, 2].set_title(f"Disk Count Rate (D={diameter.value}m)")
    # if i == len(diameters) - 1:  # Only add colorbar to last row
    #     plt.colorbar(im2, ax=axs_disk[i, 2], label="Count Rate")

    # Plot post-processed image
    im3 = axs_disk[i, 3].imshow(
        processed_ds["processed_image(coro)"].squeeze(),
        norm=colors.LogNorm(),
        origin="lower",
    )
    axs_disk[i, 3].set_title(f"Processed Image (D={diameter.value}m)")

fig_disk.suptitle(f"Disk Analysis ({coronagraph_dir1.stem})")
fig_disk.tight_layout()
fig_disk.savefig(f"output/{coronagraph_dir1.stem}_disk_analysis_diameters.png")
plt.show()
breakpoint()

# Continue with planet analysis
n_planets = len(system.planets)
planets = copy.deepcopy(system.planets)
xy_vals = (
    np.array(
        [
            [-0.34954775, -7.45104138],
            [-18.47159373, -12.52078455],
            [-12.26322495, 10.24655195],
            [37.00591501, 22.20057023],
            [83.21488627, 103.98671726],
            [116.69334482, 196.00530078],
            [490.9797864, 44.65935628],
        ]
    )
    + 128
)
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for planet, ax, i in zip(planets, axs.flatten(), np.arange(n_planets)):
    system.planets = [planet]

    # Loop over coronagraphs and simulate observations
    obs = Observation(coro, system, observing_scenario, settings)
    obs.create_count_rates()
    raw_ds = obs.count_photons()
    processed_ds = post.run(raw_ds)
    ax.imshow(
        processed_ds["planet(coro)"].squeeze(),
        norm=colors.LogNorm(),
        origin="lower",
    )
    pix_loc = xy_vals[i]
    ax.set_title(f"Planet {i} (x={pix_loc[0]:.0f}, y={pix_loc[1]:.0f})")
    ax.scatter(*pix_loc, color="red")
ax = axs.flatten()[-1]
system.planets = planets
obs = Observation(coro, system, observing_scenario, settings)
obs.create_count_rates()
raw_ds = obs.count_photons()
processed_ds = post.run(raw_ds)
ax.imshow(
    processed_ds["processed_image(coro)"].squeeze(),
    norm=colors.LogNorm(),
    origin="lower",
)
ax.set_title("All planets")

fig.tight_layout()
fig.savefig(f"output/{coronagraph_dir1.stem}_planets.png")
plt.show()
