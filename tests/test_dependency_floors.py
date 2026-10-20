"""The packaging metadata must keep old and new library releases apart.

The sky convention changed across orbix 3 and skyscapes 2.1: source offsets
are (east, north, away), and the detector reflects east. An install that
mixes an older orbix or skyscapes with this release would place sources on
the wrong side of the star without raising an error. These tests read
pyproject.toml and assert the minimum versions.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
FLOORS = {"orbix": Version("3.0.0"), "skyscapes": Version("2.1.0")}


def _all_requirements():
    """Return (where, Requirement) for every dependency the metadata declares."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    for name, reqs in project.get("optional-dependencies", {}).items():
        groups[f"extra {name}"] = reqs
    return [(w, Requirement(t)) for w, reqs in groups.items() for t in reqs]


def _lower_bound(requirement):
    """Return the smallest version a requirement admits, or None if unbounded."""
    bounds = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in (">=", "==", "~=", ">")
    ]
    return max(bounds) if bounds else None


def test_core_dependencies_carry_the_convention_floors():
    """Orbix and skyscapes are core dependencies with the convention floors."""
    core = {
        req.name: req for where, req in _all_requirements() if where == "dependencies"
    }
    for name, floor in FLOORS.items():
        assert name in core, f"{name} is not a core dependency"
        bound = _lower_bound(core[name])
        assert bound is not None and bound >= floor, (
            f"{core[name]} admits a {name} older than {floor}"
        )


def test_no_extra_loosens_a_floor():
    """If an extra names orbix or skyscapes, it keeps the same floor."""
    for where, req in _all_requirements():
        if req.name in FLOORS:
            bound = _lower_bound(req)
            assert bound is not None and bound >= FLOORS[req.name], (
                f"{where}: {req} admits a {req.name} older than {FLOORS[req.name]}"
            )
