# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

INNER_TEST = """\
import os


def test_writes_model_files_and_changes_directory():
    open("stray.eq", "wb").close()
    os.chdir(os.path.join("..", "elsewhere"))
    open("other.eq", "wb").close()
"""


def test_the_guard_checks_the_directory_the_session_started_in(tmp_path) -> None:
    """A session started in ``start``, which already holds ``old.eq``, runs a test
    that writes ``stray.eq`` there, then changes to ``elsewhere`` without changing
    back and writes ``other.eq``. The guard fails the session at teardown and
    names ``stray.eq`` alone."""
    start, elsewhere = tmp_path / "start", tmp_path / "elsewhere"
    start.mkdir()
    elsewhere.mkdir()
    (start / "old.eq").write_bytes(b"")
    (start / "test_inner.py").write_text(INNER_TEST)
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
        cwd=start,
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
        "test_writes_model_files_and_changes_directory": [
            (
                "error",
                'failed on teardown with "AssertionError: tests wrote model files '
                f"into {os.path.realpath(start)}: ['stray.eq']",
            )
        ]
    }, output
