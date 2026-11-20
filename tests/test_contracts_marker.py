# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest
from conftest import CONTRACTS_SKIP_REASON


def test_only_python_o_skips_the_contracts_tests(
    request: pytest.FixtureRequest,
) -> None:
    """Every collected test marked ``contracts`` is skipped under python -O and none
    is skipped otherwise; no unmarked test carries that skip."""
    items = request.session.items
    marked = [
        item for item in items if item.get_closest_marker("contracts") is not None
    ]
    skipped = [item for item in marked if item.get_closest_marker("skip") is not None]
    assert skipped == ([] if __debug__ else marked)
    unmarked_skipped = [
        item
        for item in items
        if item.get_closest_marker("contracts") is None
        and any(
            mark.kwargs.get("reason") == CONTRACTS_SKIP_REASON
            for mark in item.iter_markers("skip")
        )
    ]
    assert unmarked_skipped == []


def test_python_o_skips_a_marked_test_and_runs_an_unmarked_one(tmp_path) -> None:
    """A run under PYTHONOPTIMIZE=1 of one marked test and one unmarked,
    parametrized test skips the first and passes the second."""
    report = tmp_path / "report.xml"
    env = {**os.environ, "PYTHONOPTIMIZE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-o",
            "addopts=",
            f"--junit-xml={report}",
            "tests/test_equine_protonet.py::"
            "test_an_ood_distance_of_the_wrong_type_is_refused_by_beartype",
            "tests/test_utils.py::test_cholesky_with_jitter_stops_at_its_cap[dtype0]",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout[-2000:] + result.stderr[-2000:]
    assert report.exists(), output
    outcomes = {
        case.get("name"): [
            (child.tag, child.get("type"), child.get("message")) for child in case
        ]
        for case in ElementTree.parse(report).getroot().iter("testcase")
    }
    assert outcomes == {
        "test_an_ood_distance_of_the_wrong_type_is_refused_by_beartype": [
            ("skipped", "pytest.skip", CONTRACTS_SKIP_REASON)
        ],
        "test_cholesky_with_jitter_stops_at_its_cap[dtype0]": [],
    }, output
