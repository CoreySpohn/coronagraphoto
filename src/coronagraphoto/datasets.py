"""Data management utilities for coronagraphoto.

Fetches example data files for testing and documentation. The example
coronagraph YIP is delegated to ``yippy.fetch_yip`` so there is exactly
one canonical YIP hosting location across the workspace; example
ExoVista scenes still come from the coronalyze raw data folder.

Example:
    >>> from coronagraphoto.datasets import fetch_coronagraph, fetch_scene
    >>> coro_path = fetch_coronagraph()
    >>> scene_path = fetch_scene()
"""

import pooch
from pooch import Unzip
from yippy import fetch_yip

REGISTRY = {
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
}

PIKACHU = pooch.create(
    path=pooch.os_cache("coronagraphoto"),
    base_url="https://github.com/CoreySpohn/coronalyze/raw/main/data/",
    registry=REGISTRY,
)


def fetch_coronagraph() -> str:
    """Fetch and unpack the example coronagraph YIP.

    Delegates to ``yippy.fetch_yip("eac1_aavc_2d")`` so coronagraphoto
    reuses yippy's canonical YIP cache instead of duplicating hosting.

    Returns:
        Path to the coronagraph directory.
    """
    return fetch_yip("eac1_aavc_2d")


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
