# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from conftest import available_devices

INNER_TEST = """\
import torch

STATES = []


def rng_state():
    if DEVICE == "cpu":
        return torch.get_rng_state()
    return torch.get_device_module(DEVICE).get_rng_state()


def test_reseeds():
    STATES.append(rng_state())
    torch.manual_seed(1234)


def test_starts_from_the_first_tests_state():
    assert torch.equal(rng_state(), STATES[0]), "the RNG state was not restored"
"""


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(device, marks=[] if device == "cpu" else [pytest.mark.accelerator])
        for device in available_devices()
    ],
)
def test_the_fixture_restores_the_rng_state_a_test_reseeds(device, tmp_path) -> None:
    """An inner session runs a test that reseeds torch, then a test that reads
    ``device``'s RNG state. The second test sees the state the first one
    started from."""
    (tmp_path / "test_inner.py").write_text(f"DEVICE = {device!r}\n" + INNER_TEST)
    report = tmp_path / "report.xml"
    tests = str(Path(__file__).resolve().parent)
    env = {**os.environ}
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [tests, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "conftest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-o",
            "addopts=",
            f"--junit-xml={report}",
            "test_inner.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout[-2000:] + result.stderr[-2000:]
    assert report.exists(), output
    outcomes = {
        case.get("name"): [
            (child.tag, child.get("message", "").split("\n")[0]) for child in case
        ]
        for case in ElementTree.parse(report).getroot().iter("testcase")
    }
    assert outcomes == {
        "test_reseeds": [],
        "test_starts_from_the_first_tests_state": [],
    }, output
