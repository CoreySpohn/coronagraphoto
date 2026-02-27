"""Data management utilities for coronagraphoto.

This module provides utilities for fetching example data files for testing
and documentation. It reuses the same pooch registry as coronalyze to avoid
duplication.

Example:
    >>> from coronagraphoto.datasets import fetch_coronagraph, fetch_scene
    >>> coro_path = fetch_coronagraph()
    >>> scene_path = fetch_scene()
"""

import pooch
from pooch import Unzip

REGISTRY = {
    "coronagraphs.zip": "md5:1537f41c20cb10170537a7d4e89f64b2",
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
}

PIKACHU = pooch.create(
    path=pooch.os_cache("coronagraphoto"),
    base_url="https://github.com/CoreySpohn/coronalyze/raw/main/data/",
    registry=REGISTRY,
)


def fetch_coronagraph() -> str:
    """Fetch and unpack example coronagraph data.

    Downloads the eac1_aavc_512 coronagraph (apodized vortex) for use
    with yippy and coronagraphoto.

    Returns:
        Path to the coronagraph directory.
    """
    PIKACHU.fetch("coronagraphs.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "coronagraphs.zip.unzip" / "coronagraphs" / "eac1_aavc_512"
    )


def fetch_scene() -> str:
    """Fetch and unpack example ExoVista scene data.

    Downloads a modified Solar System scene for demonstration.

    Returns:
        Path to the ExoVista FITS file.
    """
    PIKACHU.fetch("scenes.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "scenes.zip.unzip" / "scenes" / "solar_system_mod.fits"
    )


def fetch_all() -> tuple[str, str]:
    """Fetch all example data.

    Returns:
        Tuple of (coronagraph_path, scene_path).
    """
    return fetch_coronagraph(), fetch_scene()
