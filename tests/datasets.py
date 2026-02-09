"""Data management utilities for coronagraphoto tests.

This module provides utilities for fetching example data files for testing.
It reuses the same pooch registry as coronalyze to avoid duplication.
"""

import pooch
from pooch import Unzip

# Create a pooch registry for data files
# Uses the same data as coronalyze to avoid duplication
REGISTRY = {
    "coronagraphs.zip": "md5:1537f41c20cb10170537a7d4e89f64b2",
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
}

# Create a pooch instance for coronagraphoto test data
# Data is cached in the user's cache directory
POKE = pooch.create(
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
    POKE.fetch("coronagraphs.zip", processor=Unzip())
    return str(
        POKE.abspath / "coronagraphs.zip.unzip" / "coronagraphs" / "eac1_aavc_512"
    )


def fetch_scene() -> str:
    """Fetch and unpack example ExoVista scene data.

    Downloads a modified Solar System scene for demonstration.

    Returns:
        Path to the ExoVista FITS file.
    """
    POKE.fetch("scenes.zip", processor=Unzip())
    return str(POKE.abspath / "scenes.zip.unzip" / "scenes" / "solar_system_mod.fits")


def fetch_all() -> tuple[str, str]:
    """Fetch all example data.

    Returns:
        Tuple of (coronagraph_path, scene_path).
    """
    return fetch_coronagraph(), fetch_scene()
