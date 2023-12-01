from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from exoverses.base.planet import Planet
from exoverses.base.star import Star
from exoverses.base.system import System
from exoverses.exovista.system import ExovistaSystem
from lod_unit.lod_unit import lod, lod_eq
from scipy.ndimage import rotate, shift, zoom

from coroSims import coronagraph

coronagraph_dir = Path("input/coronagraphs/LUVOIR-B-VC6_timeseries/")
coro = coronagraph.Coronagraph(coronagraph_dir)

# Input files
scene = Path("input/scenes/999-HIP_-TYC_SUN-mv_4.83-L_1.00-d_10.00-Teff_5778.00.fits")
coronagraph_dir = Path("input/coronagraphs/LUVOIR-B-VC6_timeseries/")
# coronagraph_dir = Path(
#     "input/coronagraphs/LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb/"
# )

system = ExovistaSystem(scene)
breakpoint()
# Make plot comparing the interpolation pixel values of the planets
# to the values in the arrays
cmap = plt.get_cmap("viridis")
base_color = cmap(0.25)
interp_color = cmap(0.75)
fig, ax = plt.subplots()
for planet in system.planets:
    ev_x = planet._x_pix
    ev_y = planet._y_pix
    ev_t = planet._t
    expanded_times = Time(np.linspace(ev_t[0].jd, ev_t[-1].jd, 1000), format="jd")
    interp_x = planet._x_pix_interp(expanded_times)
    interp_y = planet._y_pix_interp(expanded_times)
    base = ax.scatter(
        ev_x, ev_y, label="ExoVista", s=1, alpha=0.25, color=base_color, zorder=2
    )
    interp = ax.plot(
        interp_x,
        interp_y,
        label="Interpolation",
        linestyle="--",
        color=interp_color,
        zorder=1,
    )
ax.set_xlabel("x [pix]")
ax.set_ylabel("y [pix]")
ax.set_title("Planet Pixel Positions")
fig.legend(handles=[base, interp[0]])
fig.savefig("results/validation/planet_pixel_positions.png", dpi=300)
plt.close(fig)

# times = Time(np.linspace(2000, 2001, 10), format="decimalyear")
# system.propagate_img(times)
coro = coronagraph.Coronagraph(coronagraph_dir)

# Remove all planets but the first
nplanet = 5
system.planets = [system.planets[nplanet]]

D = 1 * u.m
lam = 1 * u.um
obs_scenario = {
    "diameter": D,
    "wavelengths": lam,
    "times": Time(2000, format="decimalyear"),
}

# Get (x, y) in lambda/D
planet = system.planets[0]
xy_pos = u.Quantity([planet._x_pix[0], planet._y_pix[0]])
mas_pos = xy_pos * system.pixel_scale
lod_pos = mas_pos.to(lod, lod_eq(lam, D))

ang_sep_lod = np.sqrt(np.sum(lod_pos**2))
offax_psf = coro.offax_psf_interp(ang_sep_lod)
rot_angle = np.arctan2(lod_pos[1].value, lod_pos[0].value) * u.rad
offax_psf = rotate(offax_psf, -rot_angle.to(u.deg).value, reshape=False)

coords = offax_psf.shape * u.pixel
coords_lod = coords * coro.pixel_scale
pixels_in_lod = ((((offax_psf.shape[0] - 1) // 2) * u.pixel) * coro.pixel_scale).value
plt.imshow(
    offax_psf, extent=(-pixels_in_lod, pixels_in_lod, pixels_in_lod, -pixels_in_lod)
)
plt.title(
    (
        f"Off-axis PSF at planet separation {ang_sep_lod.value:.2f}"
        f"({lod_pos[0].value:.2f},{lod_pos[1].value:.2f})"
    )
)
plt.xlabel("x [lambda/D]")
plt.ylabel("y [lambda/D]")
plt.colorbar()
plt.savefig("results/validation/offax_psf.png", dpi=300)

abs_lod_pos = lod_pos + pixels_in_lod * lod
abs_pix_pos = (abs_lod_pos / coro.pixel_scale).value.astype(int)
max_loc = np.array(np.unravel_index(np.argmax(offax_psf), offax_psf.shape))
max_loc_lod = (max_loc * u.pix) * coro.pixel_scale
plt.scatter(
    [max_loc_lod[1].value - pixels_in_lod],
    [max_loc_lod[0].value - pixels_in_lod],
    marker="x",
    color="red",
    s=10,
)
plt.show()
breakpoint()
photometric_throughput = offax_psf

# offax

# Simulate observations
observations = observations.Observations(coro, system, observing_scenario)
