"""Equinox wrapper for yippy Coronagraph to enable JAX compilation.

This module provides a JAX-compatible Equinox module that wraps yippy's
Coronagraph class. The wrapper extracts the necessary data from the yippy
Coronagraph object and converts it to JAX arrays and interpax interpolators
for use in JIT-compiled code.
"""

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array
from yippy import Coronagraph as YippyCoronagraph
from yippy.offjax import OffJAX


class Coronagraph(eqx.Module):
    """JAX-compatible Equinox wrapper for yippy Coronagraph.

    This wrapper extracts data from a yippy Coronagraph instance and converts
    it to JAX-compatible formats (JAX arrays and interpax interpolators) for
    use in JIT-compiled code.

    The wrapper provides:
    - `create_psf(x, y)`: Create a single off-axis PSF at position (x, y) in lambda/D
    - `create_psfs(x_vals, y_vals)`: Create multiple PSFs (vectorized)
    - `stellar_intens(stellar_diam)`: Get stellar intensity map for a given diameter
    - `psf_datacube`: Optional 4D PSF datacube for disk simulations (shape: ny, nx, ny, nx)

    Attributes:
        pixel_scale_lod: Pixel scale in lambda/D per pixel.
        psf_shape: Shape of the PSF images (height, width).
        center_x: X center coordinate in pixels.
        center_y: Y center coordinate in pixels.
        create_psf: JAX-compiled function to create off-axis PSFs.
        create_psfs: JAX-compiled vectorized function to create multiple PSFs.
        _stellar_ln_interp: interpax CubicSpline for log-space stellar intensity interpolation.
        psf_datacube: Optional 4D PSF datacube array for disk convolution (None if not loaded).
        sky_trans: Sky transmission map array.
    """

    pixel_scale_lod: float
    psf_shape: tuple[int, int]
    center_x: float
    center_y: float
    create_psf: callable
    create_psfs: callable
    _stellar_ln_interp: interpax.CubicSpline
    psf_datacube: jnp.ndarray | None
    sky_trans: Array

    def __init__(self, yippy_coro: YippyCoronagraph, ensure_psf_datacube: bool = False):
        """Initialize the Equinox wrapper from a yippy Coronagraph instance.

        Args:
            yippy_coro: A yippy Coronagraph instance that has been fully initialized.
                Must have use_jax=True to enable JAX functionality.
            ensure_psf_datacube: If True, generate/load the PSF datacube if it doesn't exist.
                The datacube is only needed for disk simulations. Default is False.
        """
        # Extract basic properties
        self.pixel_scale_lod = float(yippy_coro.pixel_scale.value)
        self.psf_shape = tuple(map(int, yippy_coro.psf_shape))
        self.center_x = float(yippy_coro.offax.center_x.value)
        self.center_y = float(yippy_coro.offax.center_y.value)
        # Extract off-axis PSF creation functions from OffJAX
        # Check if it's the JAX version
        if not isinstance(yippy_coro.offax, OffJAX):
            raise ValueError(
                "yippy Coronagraph must be initialized with use_jax=True "
                "to use the Equinox wrapper"
            )

        # Steal the JAX-compatible functions directly
        self.create_psf = yippy_coro.offax.create_psf
        self.create_psfs = yippy_coro.offax.create_psfs

        # Extract stellar intensity data and convert to JAX/interpax
        stellar_intens = yippy_coro.stellar_intens

        # Convert stellar diameters from astropy Quantity to JAX array (lambda/D values)
        stellar_diams_lod = jnp.asarray(stellar_intens.diams.value, dtype=jnp.float32)

        # Convert PSF data to JAX array
        stellar_psfs = jnp.asarray(stellar_intens.psfs, dtype=jnp.float32)
        ln_psfs = jnp.log(stellar_psfs)
        # Create interpax interpolation for log-space stellar intensity
        self._stellar_ln_interp = interpax.CubicSpline(stellar_diams_lod, ln_psfs)

        # PSF datacube will be set by from_yippy if needed
        # Initialize as None - will be populated during wrapper creation if datacube exists
        if ensure_psf_datacube:
            # Avoid creating a copy if already the right dtype and a JAX array
            datacube = yippy_coro.psf_datacube
            if isinstance(datacube, jax.Array) and datacube.dtype == jnp.float32:
                # Already a JAX array with correct dtype - use directly
                self.psf_datacube = datacube
            else:
                # Convert to JAX array with correct dtype
                self.psf_datacube = jnp.asarray(datacube, dtype=jnp.float32)
            # Release reference in yippy to avoid duplicate storage
            yippy_coro.psf_datacube = None
        else:
            self.psf_datacube = None

        # Get the sky transmission data
        self.sky_trans = jnp.asarray(yippy_coro.sky_trans(), dtype=jnp.float32)

    def stellar_intens(self, stellar_diam_lod: float) -> Array:
        """Get stellar intensity map for a given stellar diameter.

        Args:
            stellar_diam_lod: Stellar diameter in lambda/D.

        Returns:
            JAX array of shape (height, width) containing the stellar intensity map.
        """
        # Interpolate in log space along axis 0 (diameters)
        return jnp.exp(self._stellar_ln_interp(stellar_diam_lod))


def from_yippy(
    yippy_coro: YippyCoronagraph, ensure_psf_datacube: bool = False
) -> Coronagraph:
    """Factory function to create an Equinox Coronagraph from a yippy Coronagraph.

    Args:
        yippy_coro: A fully initialized yippy Coronagraph instance.
        ensure_psf_datacube: If True, generate/load the PSF datacube if it doesn't exist.
            The datacube is only needed for disk simulations. Default is False.

    Returns:
        An Equinox Coronagraph wrapper that can be used in JIT-compiled code.

    Example:
        ```python
        from yippy import Coronagraph as YippyCoronagraph
        from coronagraphoto.optical_elements.coronagraph import from_yippy

        # Create yippy coronagraph (outside JIT)
        yippy_coro = YippyCoronagraph("path/to/coronagraph", use_jax=True)

        # Create Equinox wrapper (without datacube)
        coro = from_yippy(yippy_coro)

        # Or create with datacube if needed for disk simulations
        coro = from_yippy(yippy_coro, ensure_psf_datacube=True)

        # Now coro can be used in JIT-compiled functions
        ```
    """
    if ensure_psf_datacube:
        yippy_coro.create_psf_datacube()
    return Coronagraph(yippy_coro, ensure_psf_datacube=ensure_psf_datacube)
