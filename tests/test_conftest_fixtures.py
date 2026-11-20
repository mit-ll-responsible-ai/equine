# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT
"""The conftest fixtures that act on a whole test, run in an inner pytest session."""

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from conftest import available_devices

RNG_INNER_TEST = """\
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

STRAY_INNER_TEST = """\
import os


def test_writes_model_files_and_changes_directory():
    open("stray.eq", "wb").close()
    os.chdir(os.path.join("..", "elsewhere"))
    open("other.eq", "wb").close()
"""


def _run_inner_session(cwd: Path, report: Path) -> tuple[dict, str]:
    """Run ``test_inner.py`` in ``cwd`` in a new pytest process that loads this
    directory's conftest; return each test case's result elements as (tag, first
    line of the message), and the end of the process's output."""
    env = {**os.environ}
    env.pop("PYTEST_ADDOPTS", None)
    tests = str(Path(__file__).resolve().parent)
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
        cwd=cwd,
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
    return outcomes, output


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
    (tmp_path / "test_inner.py").write_text(f"DEVICE = {device!r}\n" + RNG_INNER_TEST)
    outcomes, output = _run_inner_session(tmp_path, tmp_path / "report.xml")
    assert outcomes == {
        "test_reseeds": [],
        "test_starts_from_the_first_tests_state": [],
    }, output


def test_the_guard_checks_the_directory_the_session_started_in(tmp_path) -> None:
    """A session started in ``start``, which already holds ``old.eq``, runs a test
    that writes ``stray.eq`` there, then changes to ``elsewhere`` without changing
    back and writes ``other.eq``. The guard fails the session at teardown and
    names ``stray.eq`` alone."""
    start, elsewhere = tmp_path / "start", tmp_path / "elsewhere"
    start.mkdir()
    elsewhere.mkdir()
    (start / "old.eq").write_bytes(b"")
    (start / "test_inner.py").write_text(STRAY_INNER_TEST)
    outcomes, output = _run_inner_session(start, tmp_path / "report.xml")
    assert outcomes == {
        "test_writes_model_files_and_changes_directory": [
            (
                "error",
                'failed on teardown with "AssertionError: tests wrote model files '
                f"into {os.path.realpath(start)}: ['stray.eq']",
            )
        ]
    }, output
