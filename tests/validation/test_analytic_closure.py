"""Closure of the analytic aperture rates against the image path's aperture record.

``jaxedith.rates`` reads four sources from tabulated performance curves of the
coronagraph, plus the warm telescope's thermal emission.
``coronagraphoto.aperture_rates`` reads the same five from images of the same
scene rendered on the detector. They share the telescope, the
scene, the detector and the aperture rule, so the gap between them is the gap
between the idealized curve read and the image. It is measured here, source by
source, against separation, and each part of it is tied to a cause by a check
that varies the cause and predicts the gap from the cause alone.

The instrument is the synthetic Gaussian YIP of the other closure tests (sigma
0.7 lambda/D, integral 0.3, a circular aperture of radius 0.7 lambda/D) with
the scene of ``test_calculator_closure``. A second YIP differs only in a
stellar leakage that falls with radius, so the leakage curve is not flat.
Every gap below is ``analytic / image - 1`` of the same rate.

Causes, as measured on this instrument:

- Band-centre aperture. The analytic read fixes the aperture and the lambda/D
  scale at the band centre, while the image renders the PSF and the stellar
  map at each wavelength node. At a node the stellar leakage differs by
  ``(node / centre)**2 - 1`` exactly (up to 8% at the ends of a 50 nm band at
  525 nm). The quadrature sum over nodes cancels the first order, and what
  remains is second order in the band width.
- Pixelization. The curves describe the aperture in the limit of infinitely
  fine pixels; the image sums pixel weights. The planet gap falls as the pixel
  pitch squared.
- Disk structure. The analytic read multiplies the disk radiance at the
  aperture centre by the aperture solid angle. The image carries the disk
  through the coronagraph, which blurs it with the PSF, and sums it over the
  aperture. Where the radiance changes across the PSF and the aperture, in
  particular across the disk's inner edge, the two differ by tens of percent.
- Curve interpolation. The curves are tabulated at the YIP's offsets (0, 1, 2
  and 3 lambda/D) and interpolated between them, with a log-linear tail
  beyond the last. The image holds the leakage at every position.
"""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    YIP_PITCH_LOD,
    YIP_SIGMA_LOD,
    closure_system,
    make_path,
    write_yip,
)
from jaxedith import rates as analytic_rates
from optixstuff import Detector, WarmEmission
from optixstuff.apertures import Circular
from optixstuff.spectral import Band
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from yippy import open_yip

import coronagraphoto as cp

pytestmark = pytest.mark.medium(
    reason="two YIP loads and a dozen aperture responses at three pixel pitches"
)

SOURCES = ("planet", "star_leakage", "local_zodi", "exozodi", "thermal")
RULE = Circular(0.7)
ROLL, CONTRAST = 15.0, 1e-9
BANDS = (Band(500.0, 550.0), Band(550.0, 600.0))
NARROW_BANDS = (Band(524.0, 526.0), Band(574.0, 576.0))
# Separations in lambda/D at 550 nm. The last curve knot is at 3 lambda/D and
# the detector edge matters beyond it; 3.4 is measured but not registered.
SEPARATIONS = (0.8, 1.0, 1.5, 2.0, 2.5, 3.0)
BEYOND = 3.4
POSITION_ANGLES = (250.0, 20.0)
# Half-width in lambda/D of the synthetic tables. Nothing outside the tables is
# rendered (the pixel reads zero and is unsupported), so the closures need the
# tables to cover every pixel of the largest field (12.8 lambda/D at 550 nm,
# 7.1 lambda/D from the centre at the 500 nm node) and every planet position.
# The band-limited reconstruction of a table that does not vanish at its edge
# is the zero-padded interpolant, whose edge ripple reaches the interior and
# falls only slowly with the reach (see LEAKAGE_RIPPLE); 16 puts it below the
# bounds registered here.
TABLE_REACH = 16
# Largest ripple of the uniform stellar map, as a relative error of the leakage
# rate, measured at the reach above (2.5e-4 per node, 1.1e-3 per 2 nm band
# against the same read without a table edge); each bound is 1.5 times it,
# rounded up.
LEAKAGE_RIPPLE_NODE, LEAKAGE_RIPPLE_BAND = 4e-4, 1.7e-3


@pytest.fixture(scope="module", autouse=True)
def band_limited_stellar_maps():
    """Resample the synthetic stellar maps band-limited, as the closures assume.

    Each closure predicts the image route from the exact synthetic profile. The
    profiles are smooth, so band-limited resampling reproduces them; the shipped
    linear default adds about 1e-2 on the 0.5 lambda/D radial table, which would
    hide the curve interpolation that test_curve_interpolation isolates.
    """
    import yippy._stellar_intens as stellar_intens
    import yippy.coronagraph as coronagraph

    with pytest.MonkeyPatch.context() as patch:
        for module in (coronagraph, stellar_intens):
            patch.setattr(module, "STELLAR_RECONSTRUCTION", "band_limited")
        yield


# Registered exozodi bounds by separation: the raw band gap, and what remains
# after the prediction with the box-widened blur kernel (measured, times 1.5
# and rounded up).
EXOZODI_GAP = {0.8: 0.40, 1.0: 0.30, 1.5: 0.16, 2.0: 0.09, 2.5: 0.05, 3.0: 0.017}
EXOZODI_RESIDUAL = {
    0.8: 7e-3,
    1.0: 2.7e-3,
    1.5: 3.3e-3,
    2.0: 7e-4,
    2.5: 9e-4,
    3.0: 2e-3,
}
# The synthetic YIP of conftest.write_yip.
YIP_SIGMA, YIP_INTEGRAL = YIP_SIGMA_LOD, 0.3
# The disk reaches the YIP as one cell-integrated flux per native cell, placed
# at the cell centre: the disk is smoothed by the cell's box (variance
# pitch**2 / 12 per axis) before the PSF, so the kernel is wider than the
# point-source PSF by this factor.
DISK_CELL_WIDENING = np.sqrt(1 + (YIP_PITCH_LOD**2 / 12) / YIP_SIGMA_LOD**2)


def radial_leakage(radius_lod):
    """Stellar leakage multiplier of the second YIP, 4 at the star and 1 far out."""
    return 1.0 + 3.0 * np.exp(-radius_lod)


class Case:
    """One instrument, band set and pixel pitch, with both readers built."""

    def __init__(self, coro, scale, system, *, pitch, n_pix, bands):
        """Build the image record and keep the curves of ``coro``."""
        base = eqx.tree_at(lambda p: p.coronagraph, make_path(), coro)
        self.coro, self.scale, self.system, self.bands = coro, scale, system, bands
        self.detector = Detector(
            shape=(n_pix, n_pix),
            pixel_scale_arcsec=pitch * scale,
            qe_curve=(jnp.array([500.0, 600.0]), jnp.array([0.9, 0.7])),
            survival_fraction=0.9,
        )
        path = eqx.tree_at(lambda p: p.detector, base, self.detector)
        self.path = eqx.tree_at(
            lambda p: p.warm_emission,
            path,
            WarmEmission(290.0, 0.5),
            is_leaf=lambda x: x is None,
        )
        self.exposure = cp.build_exposure(
            system, self.path, bands, time_d=0.0, roll_deg=ROLL
        )
        response = cp.aperture_response(
            self.path,
            self.exposure.quadrature,
            rule=RULE,
            n_separations=33,
            n_angles=16,
        )
        self.record = cp.aperture_record(system, self.path, self.exposure, response)
        self.curves = coro.curves(RULE)

    @property
    def quadrature(self):
        """The nodes of the bands, shape (W, Q)."""
        return self.exposure.quadrature

    def offset(self, separation_lod, angle_deg):
        """East and north offsets in arcsec of a separation at a position angle."""
        angle = np.deg2rad(angle_deg)
        s = separation_lod * self.scale
        return s * np.sin(angle), s * np.cos(angle)

    def read(self, separation_lod, angle_deg):
        """Return (analytic, image) rates of one planet position."""
        east, north = self.offset(separation_lod, angle_deg)
        image = cp.aperture_rates(self.record, east, north, CONTRAST, time_d=0.0)
        analytic = analytic_rates(
            self.curves,
            self.system,
            self.path,
            self.quadrature,
            east,
            north,
            CONTRAST,
            time_d=0.0,
            uniform_sky=self.coro.uniform_sky_profile,
        )
        return analytic, image

    def gaps(self, separation_lod, angle_deg, *, scale_nodes=None):
        """Return the node and band gaps of every source, analytic / image - 1.

        ``scale_nodes`` maps a source to node factors multiplied onto the
        analytic rate first, for the predictions of a cause.
        """
        analytic, image = self.read(separation_lod, angle_deg)
        if scale_nodes:
            analytic = eqx.tree_at(
                lambda r: tuple(getattr(r, k) for k in scale_nodes),
                analytic,
                tuple(getattr(analytic, k) * v for k, v in scale_nodes.items()),
            )
        electrons = self.detector.electrons(analytic), self.detector.electrons(image)
        node = {
            k: np.asarray(getattr(analytic, k) / getattr(image, k) - 1) for k in SOURCES
        }
        band = {
            k: np.asarray(getattr(electrons[0], k) / getattr(electrons[1], k) - 1)
            for k in SOURCES
        }
        band["n_pix"] = np.asarray(analytic.n_pix / image.n_pix - 1)
        return node, band, analytic, image


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """Return a cached builder of cases by YIP, pixel pitch, bands and field size."""
    cache, yips = {}, {}

    def make(*, radial=False, pitch=0.1, bands=BANDS, n_pix=None):
        n_pix = round(9.6 / pitch) if n_pix is None else n_pix
        key = (radial, pitch, bands, n_pix)
        if key not in cache:
            if radial not in yips:
                path = write_yip(
                    tmp_path_factory.mktemp("analytic-closure") / "yip",
                    stellar_profile=radial_leakage if radial else None,
                    reach=TABLE_REACH,
                    half_width=TABLE_REACH,
                )
                yips[radial] = (
                    open_yip(path).prepare(rules=(RULE,)).with_scene_operator()
                )
            coro = yips[radial]
            scale = float(
                eqx.tree_at(lambda p: p.coronagraph, make_path(), coro)
                .primary.frame(550.0)
                .lod_to_arcsec(550.0)
            )
            system = closure_system(scale)
            cache[key] = Case(
                coro, scale, system, pitch=pitch, n_pix=n_pix, bands=bands
            )
        return cache[key]

    return make


def _band_centres(bands):
    return np.array([0.5 * (b.low_nm + b.high_nm) for b in bands])


def chromatic_leakage_gap(case):
    """Node gap of the stellar leakage that the band-centre aperture predicts.

    The image path puts the stellar map at (node / 550 nm) times its lambda/D
    scale, so the leakage per fixed aperture goes as 1 / node**2 while the
    analytic read holds the band-centre scale: analytic / image - 1 =
    (node / centre)**2 - 1, for a leakage that is uniform in separation.
    """
    nodes = np.asarray(case.quadrature.nodes_nm)
    return (nodes / _band_centres(case.bands)[:, None]) ** 2 - 1


def chromatic_planet_factor(case, tau_curve):
    """Node factor that moves the analytic planet to the rendered Gaussian PSF.

    The aperture is fixed at 0.7 lambda/D of the band centre and the PSF
    (sigma 0.7, integral 0.3 at 550 nm) scales with the node, so the planet
    fraction is the Gaussian's enclosed fraction at radius ratio centre / node.
    """
    nodes = np.asarray(case.quadrature.nodes_nm)
    ratio = _band_centres(case.bands)[:, None] / nodes
    enclosed = YIP_INTEGRAL * (
        1 - np.exp(-0.5 * (RULE.radius_lod / YIP_SIGMA * ratio) ** 2)
    )
    return enclosed / tau_curve


def disk_blur_factors(case, separation_lod, angle_deg, *, step=0.05, sigma_scale=1.0):
    """Node ratios image / analytic of the exozodi from the disk alone.

    Returns (blurred, unblurred). The image sums the disk blurred by the
    node's PSF (Gaussian, sigma 0.7 lambda/D at 550 nm, scaled with the node)
    over the aperture; the analytic read takes the radiance at the aperture
    centre. ``unblurred`` is the aperture mean of the disk without the blur.
    Both come from ``Scene.photon_radiance_at`` on a grid of ``step``
    lambda/D, one evaluation per node.
    """
    east, north = case.offset(separation_lod, angle_deg)
    nodes = np.asarray(case.quadrature.nodes_nm)
    centres = _band_centres(case.bands)
    h = step * case.scale
    reach = (
        RULE.radius_lod * centres.max() / 550.0
        + 4 * sigma_scale * YIP_SIGMA * nodes.max() / 550.0
    )
    half = int(np.ceil(reach * case.scale / h))
    g = h * np.arange(-half, half + 1)
    e_grid, n_grid = np.meshgrid(east + g, north + g)

    @jax.jit
    def radiance(node, e, n):
        return jax.vmap(
            lambda e, n: case.system.photon_radiance_at(node[None], 0.0, e, n)[0]
        )(e, n)

    blurred, unblurred = np.zeros(nodes.shape), np.zeros(nodes.shape)
    for w, centre in enumerate(centres):
        radius = RULE.radius_lod * case.scale * centre / 550.0
        inside = np.hypot(*np.meshgrid(g, g)) <= radius
        for q, node in enumerate(nodes[w]):
            sigma = sigma_scale * YIP_SIGMA * case.scale * node / 550.0
            disk = np.asarray(
                radiance(node, jnp.asarray(e_grid.ravel()), jnp.asarray(n_grid.ravel()))
            ).reshape(e_grid.shape)
            unblurred[w, q] = disk[inside].mean() / disk[half, half]
            blurred[w, q] = (
                gaussian_filter(disk, sigma / h, mode="nearest")[inside].mean()
                / disk[half, half]
            )
    return blurred, unblurred


def leakage_interpolation_gap(case, separation_lod):
    """Per-band gap of the leakage curve read from its knots alone (radial YIP).

    The tabulated values are the exact aperture means of the radial profile at
    the knots 0, 1, 2 and 3 lambda/D. The curve is PCHIP in log space between
    them and log-linear with the last secant slope beyond. The prediction is
    that read divided by the exact aperture mean at the band-centre separation.
    """

    def aperture_mean(s, h=0.01):
        g = h * np.arange(-round(RULE.radius_lod / h), round(RULE.radius_lod / h) + 1)
        x, y = np.meshgrid(g + s, g)
        inside = np.hypot(*np.meshgrid(g, g)) <= RULE.radius_lod
        return radial_leakage(np.hypot(x, y))[inside].mean()

    knots = np.asarray(case.curves.separation_lod)
    values = np.array([aperture_mean(k) for k in knots])
    spline = PchipInterpolator(knots, np.log(values))
    tail = np.log(values[-1] / values[-2])
    gap = []
    for centre in _band_centres(case.bands):
        s = separation_lod * 550.0 / centre
        read = (
            np.exp(spline(s))
            if s <= knots[-1]
            else values[-1] * np.exp(tail * (s - knots[-1]))
        )
        gap.append(float(read / aperture_mean(s) - 1))
    return np.array(gap)


@pytest.fixture(scope="module")
def table(build):
    """Gaps of every source on the uniform YIP and on the radial-leakage YIP."""
    rows = {}
    for radial in (False, True):
        case = build(radial=radial)
        for angle in POSITION_ANGLES:
            for s in (*SEPARATIONS, BEYOND):
                node, band, *_ = case.gaps(s, angle)
                rows[radial, angle, s] = dict(node=node, band=band)
    print("closure table (band gaps, analytic / image - 1)")
    for (radial, angle, s), row in rows.items():
        print(
            "radial" if radial else "uniform",
            angle,
            s,
            {k: np.round(v, 5).tolist() for k, v in row["band"].items()},
        )
    return rows


def _worst(table, source, *, radial, level="band", separations=SEPARATIONS):
    return max(
        float(np.abs(table[radial, a, s][level][source]).max())
        for a in POSITION_ANGLES
        for s in separations
    )


@vv.case("etc-count-rates", "code-verification", srq="analytic against image rates")
def test_registered_bounds(table):
    """Fails if an analytic band rate leaves its measured gap to the image rate.

    Bounds are on the electron rate of each band (the quantity S/N is built
    from) for separations 0.8 to 3 lambda/D at two position angles, on
    0.1 lambda/D pixels and 50 nm bands, on the uniform YIP unless stated.
    Basis: measured on this instrument; each bound is 1.5 times the largest
    measured gap, rounded up.

    - planet: node gaps reach 6.7% (band-centre aperture) and the quadrature
      sum leaves 3.6e-4 at most: the second-order chromatic term (-2.9e-3)
      and the pixelization (+3.0e-3) nearly cancel. Bound 6e-4, which holds
      only at this pixel pitch (see the pixelization test).
    - star_leakage: 4.1e-3, the second-order chromatic term. Bound 6.2e-3. On
      the radial-leakage YIP the curve interpolation adds and the worst is
      3.4e-2 at 0.8 lambda/D: bound 5.1e-2.
    - local_zodi, thermal and n_pix: the zodi, the sky transmission and the
      warm emission (290 K, emissivity 0.5) are uniform and both readers use
      the same aperture, so only rounding remains: 1e-9.
    - exozodi: the raw gap falls from 36% at 0.8 lambda/D to 0.9% at 3
      (``EXOZODI_GAP``); ``test_disk_structure`` explains it.

    Near the detector edge (3.4 lambda/D) the gaps are larger (exozodi 7.3e-2)
    and are not registered; ``test_curve_interpolation`` and
    ``test_detector_edge`` name why.
    """
    assert _worst(table, "planet", radial=False) < 6e-4
    assert _worst(table, "star_leakage", radial=False) < 6.2e-3
    assert _worst(table, "star_leakage", radial=True) < 5.1e-2
    for radial in (False, True):
        assert _worst(table, "local_zodi", radial=radial) < 1e-9
        assert _worst(table, "thermal", radial=radial) < 1e-9
        assert _worst(table, "n_pix", radial=radial) < 1e-9
    for s in SEPARATIONS:
        for angle in POSITION_ANGLES:
            gap = np.abs(table[False, angle, s]["band"]["exozodi"])
            assert np.all(gap < EXOZODI_GAP[s]), (s, angle)
    # The tabulated sources are the same across the two YIPs.
    for angle in POSITION_ANGLES:
        for s in SEPARATIONS:
            for source in ("planet", "exozodi", "local_zodi", "thermal"):
                np.testing.assert_allclose(
                    table[True, angle, s]["band"][source],
                    table[False, angle, s]["band"][source],
                    atol=1e-12,
                )


@pytest.fixture(scope="module")
def causes(build):
    """Evidence for each cause, from cases that vary one thing."""
    out = {}
    wide, narrow = build(), build(bands=NARROW_BANDS)
    out["node_leakage"] = wide.gaps(2.3, 250.0)[0]["star_leakage"]
    out["node_leakage_law"] = chromatic_leakage_gap(wide)
    out["band_leakage"] = {
        "wide": wide.gaps(2.3, 250.0)[1]["star_leakage"],
        "narrow": narrow.gaps(2.3, 250.0)[1]["star_leakage"],
    }
    # Pixelization: planet gap of a 2 nm band, for three pixel pitches.
    pitches = {0.2: 48, 0.1: 96, 0.05: 96}
    out["pixel"] = {}
    for pitch, n_pix in pitches.items():
        case = build(pitch=pitch, bands=NARROW_BANDS, n_pix=n_pix)
        out["pixel"][pitch] = dict(gap=case.gaps(1.0, 250.0)[1]["planet"])
    # Chromatic planet gap in the 50 nm bands, with the pixels made fine.
    case = build(pitch=0.1)
    tau = float(np.asarray(case.curves.throughput)[0])
    factor = chromatic_planet_factor(case, tau)
    node, band, *_ = case.gaps(1.0, 250.0)
    out["planet_chromatic"] = dict(
        node=node["planet"],
        predicted_node=1 / factor - 1,
        band=band["planet"],
        band_after=case.gaps(1.0, 250.0, scale_nodes={"planet": factor})[1]["planet"],
    )
    # Disk structure.
    out["disk"] = {}
    for s in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0):
        node, band, *_ = wide.gaps(s, 250.0)
        plain, raw = disk_blur_factors(wide, s, 250.0)
        boxed, _ = disk_blur_factors(wide, s, 250.0, sigma_scale=DISK_CELL_WIDENING)

        def left(factor, s=s):
            return wide.gaps(s, 250.0, scale_nodes={"exozodi": factor})[1]["exozodi"]

        out["disk"][s] = dict(
            gap=band["exozodi"],
            residual=left(boxed),
            plain=left(plain),
            unblurred=left(raw),
        )
    tau = float(np.asarray(wide.curves.throughput)[0])
    ideal = YIP_INTEGRAL * (1 - np.exp(-0.5 * (RULE.radius_lod / YIP_SIGMA) ** 2))
    out["point_source"] = dict(curves=tau, ideal=ideal)
    # Curve interpolation: radial YIP in 2 nm bands, uniform YIP as the control.
    out["interp"] = {}
    radial, uniform = build(radial=True, bands=NARROW_BANDS), build(bands=NARROW_BANDS)
    for s in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, BEYOND):
        out["interp"][s] = dict(
            gap=radial.gaps(s, 250.0)[1]["star_leakage"],
            predicted=leakage_interpolation_gap(radial, s),
            control=uniform.gaps(s, 250.0)[1]["star_leakage"],
        )
    # Detector edge: the exozodi at 3.4 lambda/D on a 9.6 and a 12.8 lambda/D field.
    out["edge"] = {
        n_pix: [
            build(n_pix=n_pix).gaps(BEYOND, angle)[1]["exozodi"]
            for angle in POSITION_ANGLES
        ]
        for n_pix in (96, 128)
    }
    print("causes", out)
    return out


@vv.case("etc-count-rates", "code-verification", srq="analytic against image rates")
def test_band_centre_aperture(causes):
    """Fails if the chromatic gap is not the band-centre aperture.

    The stellar leakage of a uniform curve differs at a node by exactly
    ``(node / centre)**2 - 1``; the planet by the enclosed fraction of the
    node's PSF in the band-centre aperture, up to the pixelization. Basis:
    measured; the leakage node gaps equal the law to 2.5e-4 (the product of
    the same terms plus the edge ripple of the band-limited reconstruction of
    the uniform map, ``LEAKAGE_RIPPLE_NODE``) and the planet node gaps differ from
    the pixel-free law by 3.2e-3 at most, the pixelization of the next test.
    Narrowing the band from 50 to 2 nm takes the band leakage gap from 4.1e-3
    to the ripple floor (at most 1.1e-3), second order in the width as the
    quadrature sum cancels the first.
    """
    c = causes
    np.testing.assert_allclose(
        c["node_leakage"], c["node_leakage_law"], atol=LEAKAGE_RIPPLE_NODE
    )
    planet = c["planet_chromatic"]
    assert np.abs(planet["node"] - planet["predicted_node"]).max() < 3.2e-3
    assert np.abs(planet["predicted_node"]).max() > 5e-2
    assert np.abs(c["band_leakage"]["wide"]).min() > 3.5e-3
    assert np.abs(c["band_leakage"]["narrow"]).max() < LEAKAGE_RIPPLE_BAND


@vv.case("etc-count-rates", "code-verification", srq="analytic against image rates")
def test_pixelization(causes):
    """Fails if the planet gap of a narrow band does not follow the pixel pitch.

    A 2 nm band leaves no chromatic gap, so the planet gap is the image's
    pixel sum against the curve's idealized circle. Basis: measured
    1.1e-2, 2.8e-3 and 4.5e-4 at 0.2, 0.1 and 0.05 lambda/D pixels (bands
    averaged), a fall of at least 3.5 per halving of the pitch.
    """
    gap = {p: float(np.abs(v["gap"]).mean()) for p, v in causes["pixel"].items()}
    assert gap[0.2] / gap[0.1] > 3.5 and gap[0.1] / gap[0.05] > 3.5
    assert 8e-3 < gap[0.2] < 1.4e-2
    chromatic = causes["planet_chromatic"]
    # With the chromatic law applied the 50 nm gap is the pixelization again.
    assert np.abs(chromatic["band_after"] - causes["pixel"][0.1]["gap"]).max() < 6e-4


@vv.case("etc-count-rates", "code-verification", srq="zodi and exozodi rates")
def test_disk_structure(causes):
    """Fails if the exozodi gap is not the blurred disk read across the aperture.

    Basis: the prediction blurs the node's disk radiance with a Gaussian
    kernel (sigma 0.7 lambda/D scaled with the node, widened by
    ``DISK_CELL_WIDENING``, 1.021) and averages it over the aperture, from
    ``photon_radiance_at`` on a 0.05 lambda/D grid (0.1 is not converged at
    the inner edge: it moves the 0.8 lambda/D residual by 4e-3). The widening
    is computed from the YIP pitch alone, not fitted. The image feeds the YIP
    the disk's cell integral on its 0.5 lambda/D source grid, built from 4x4
    sub-samples per cell; the operator places each cell's flux at the cell
    centre, which smooths the disk by the cell's box. With the box the
    prediction removes at least 90% of the band gap from 0.8 to 2.5 lambda/D
    and leaves 4.6e-3, 1.8e-3, 2.2e-3, 4.1e-4, 5.5e-4 and 1.2e-3 at 0.8, 1.0,
    1.5, 2.0, 2.5 and 3.0 (bounds ``EXOZODI_RESIDUAL``, 1.5 times the
    measured). The bare point-source kernel leaves 8.4e-3 and 9.5e-3 at 0.8
    and 1.0, so the test sees the widening; without any blur (aperture mean of
    the raw disk) the prediction removes less than a fifth of the gap at 0.8
    and 1.0 lambda/D, where the disk's inner edge lies within the aperture
    and PSF. The point-source path is not widened: the curves' planet
    throughput equals the ideal Gaussian enclosed fraction (0.118011 against
    0.118041).
    """
    for s, row in causes["disk"].items():
        gap, left = np.abs(row["gap"]), np.abs(row["residual"])
        assert np.all(gap < EXOZODI_GAP[s]), s
        assert np.all(left < EXOZODI_RESIDUAL[s]), s
        if s <= 2.5:
            assert np.all(left < 0.1 * gap), s
        if s <= 1.0:
            assert np.all(np.abs(row["unblurred"]) > 0.8 * gap), s
    for s in (0.8, 1.0):
        row = causes["disk"][s]
        # The cell box is what the test sees: the bare PSF kernel does worse.
        assert np.abs(row["plain"]).max() > 1.5 * np.abs(row["residual"]).max(), s
    point = causes["point_source"]
    assert abs(point["curves"] / point["ideal"] - 1) < 5e-4


@vv.case("etc-count-rates", "code-verification", srq="analytic against image rates")
def test_curve_interpolation(causes):
    """Fails if the leakage gap of a varying curve is not the curve's interpolation.

    On the radial-leakage YIP a 2 nm band has no chromatic gap. The gap is the
    PCHIP-in-log read of the curve between its knots (every lambda/D out to
    the table reach) against the exact aperture mean of the leakage profile.
    The curves end at the table edge, beyond the field, so the log-linear tail
    past the last knot is not read: nothing outside the tables is rendered, and
    the 5.4e-2 gap that tail had on the earlier 3 lambda/D tables is not
    registered. Basis: measured 3.5e-2 at the largest (0.8 lambda/D), and the
    prediction from the knots and the profile alone agrees with it to 3e-3
    (4e-3 bound); on the uniform
    YIP the same read has a gap of at most 1.1e-3, the edge ripple of the
    zero-padded band-limited reconstruction (``LEAKAGE_RIPPLE_BAND``).
    """
    rows = causes["interp"]
    for s, row in rows.items():
        assert np.abs(row["gap"] - row["predicted"]).max() < 4e-3, s
        assert np.abs(row["control"]).max() < LEAKAGE_RIPPLE_BAND, s
    assert max(np.abs(r["gap"]).max() for r in rows.values()) > 3e-2
    assert np.abs(rows[1.5]["gap"]).min() > 1e-2


@vv.case("etc-count-rates", "code-verification", srq="analytic against image rates")
def test_detector_edge(causes):
    """Fails if the 3.4 lambda/D exozodi gap depends on the detector extent.

    Disk light from sky beyond the detector edge scatters onto it, so the
    rendered exozodi must not change with the field. The gap at 3.4 lambda/D
    on a 9.6 and a 12.8 lambda/D field agrees at both position angles.
    Basis: the radiance is evaluated on the YIP's native source grid, the same
    sources in both fields, so the gaps agree to 1e-9 (measured 0). With the
    radiance evaluated on the detector grid the angle differences were 3.6e-2
    and 4.8e-2 on the small field and 4.2e-3 on the large one.
    """
    small, large = causes["edge"][96], causes["edge"][128]
    np.testing.assert_allclose(small, large, rtol=0, atol=1e-9)
