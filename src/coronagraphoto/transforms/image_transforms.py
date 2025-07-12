"""Image transformation utilities."""

import functools
import math

import jax
import jax.numpy as jnp

from coronagraphoto.transforms.map_coordinates import map_coordinates


@functools.partial(jax.jit, static_argnames=["shape_tgt", "order"])
def flux_conserving_affine(
    f_src,  # (ny_src, nx_src) integrated-flux image
    pixscale_src,  # length units per source pixel
    pixscale_tgt,  # length units per target pixel
    shape_tgt,  # (ny_tgt, nx_tgt)
    rotation_deg=0.0,  # +CCW rotation of *source* into *target*
    order=3,
):
    """Re-sample *f_src* onto the target grid while conserving total flux."""
    ny_src, nx_src = f_src.shape
    ny_tgt, nx_tgt = shape_tgt

    # 1. Surface brightness (flux per unit area)
    s_src = f_src / (pixscale_src**2)

    # 2. Affine matrix (TARGET pixel centres → SOURCE coordinates)
    theta = jnp.deg2rad(rotation_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    scale = pixscale_tgt / pixscale_src
    a_mat = jnp.array(
        [
            [scale * cos_theta, scale * sin_theta],
            [-scale * sin_theta, scale * cos_theta],
        ]
    )

    c_src = jnp.array([(ny_src - 1) / 2.0, (nx_src - 1) / 2.0])
    c_tgt = jnp.array([(ny_tgt - 1) / 2.0, (nx_tgt - 1) / 2.0])
    offset = c_src - a_mat @ c_tgt

    # 3. Grid of TARGET pixel centres
    y_coords = jnp.arange(ny_tgt, dtype=jnp.float64)
    x_coords = jnp.arange(nx_tgt, dtype=jnp.float64)
    y_tgt, x_tgt = jnp.meshgrid(y_coords, x_coords, indexing="ij")
    coords = jnp.stack([y_tgt, x_tgt], axis=0)  # (2, ny_tgt, nx_tgt)
    coords_src = (a_mat @ coords.reshape(2, -1) + offset[:, None]).reshape(coords.shape)

    # 4. Interpolate surface brightness
    s_tgt = map_coordinates(
        s_src, [coords_src[0], coords_src[1]], order=order, mode="constant", cval=0.0
    )

    # 5. Back to integrated flux per target pixel
    return s_tgt * (pixscale_tgt**2)
