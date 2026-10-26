import importlib.metadata
import os
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.mark.skipif(
    not os.environ.get("EQUINE_CHECK_FLOORS"),
    reason="only meaningful in the floors tox environment",
)
def test_installed_versions_are_the_declared_floors() -> None:
    """Every runtime and ``tests`` dependency is installed at the highest ``>=``
    floor among its requirements whose markers apply here."""
    with open(PYPROJECT, "rb") as f:
        project = tomllib.load(f)["project"]
    floors: dict[str, Version] = {}
    for line in project["dependencies"] + project["optional-dependencies"]["tests"]:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        lows = [Version(s.version) for s in requirement.specifier if s.operator == ">="]
        assert lows, f"{requirement.name} has no >= floor in pyproject.toml"
        floor = max(lows)
        floors[requirement.name] = max(floor, floors.get(requirement.name, floor))
    mismatches = []
    for name, floor in sorted(floors.items()):
        installed = Version(Version(importlib.metadata.version(name)).public)
        if installed != floor:
            mismatches.append(f"{name} {installed} (floor {floor})")
    assert not mismatches, "not at the floor: " + ", ".join(mismatches)
