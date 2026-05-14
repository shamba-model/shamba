"""Tests for monte_carlo/runner.py (Issue 3).

Covers:
- run_monte_carlo returns a list of length n_samples
- When no distribution_dict and zero climate/soil uncertainty, all results are identical
- handle_intervention is called with soil_override set (not None) on every call
- summarise_mc_results: correct column names, array shapes, and numeric values
- write_mc_summary_csv: produces a readable CSV with correct headers and year column
"""

import csv
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from model.monte_carlo.runner import run_monte_carlo, summarise_mc_results, write_mc_summary_csv


class _InProcessExecutor:
    """Drop-in replacement for ProcessPoolExecutor that runs in the same process.

    Used in tests so that mock patches on handle_intervention are visible to
    the worker function, which would otherwise run in a separate spawned process
    and not inherit the patch.
    """
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, iterable):
        return [fn(item) for item in iterable]


# ---------------------------------------------------------------------------
# Helpers — minimal fake objects matching the interfaces sampler.py expects
# ---------------------------------------------------------------------------

class FakeSoilParams:
    def __init__(self):
        self.Cy0 = 5.0
        self.clay = 30.0
        self.depth = 30.0
        self.Ceq = 10.0
        self.iom = 1.0
        self.Cy0_q05 = 5.0
        self.Cy0_q95 = 5.0
        self.clay_q05 = 30.0
        self.clay_q95 = 30.0


class FakeClimateData:
    def __init__(self):
        self.temperature = np.array([20.0] * 12)
        self.rain = np.array([80.0] * 12)
        self.evaporation = np.array([50.0] * 12)
        self.temperature_std = np.zeros(12)
        self.rain_std = np.zeros(12)
        self.evaporation_std = np.zeros(12)


BASE_INPUT = {
    "temp": np.array([20.0] * 12),
    "rain": np.array([80.0] * 12),
    "evap": np.array([50.0] * 12),
    "crop_proj_yd1": np.array([5.0, 6.0, 7.0]),
}

FIXED_RESULT = object()  # sentinel — any unique value


def fake_forward(_soil, _climate):
    return MagicMock()


def fake_inverse(_soil, _climate):
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
#
# Each test patches two names in the runner module:
#
#   handle_intervention — replaced with a MagicMock so tests never execute the
#       real model (which requires a fully-formed input dictionary).  The mock
#       lets us inspect call count and keyword arguments without caring about
#       return values.
#
#   concurrent.futures.ProcessPoolExecutor — replaced with _InProcessExecutor
#       (defined above).  patch() only modifies the current process's memory,
#       so a real ProcessPoolExecutor would spawn child processes that import
#       runner.py fresh and never see the handle_intervention mock.
#       _InProcessExecutor runs map() as a plain list comprehension in the same
#       process, keeping the mock visible to every sample call.
# ---------------------------------------------------------------------------

def test_run_monte_carlo_returns_n_samples():
    """Result list has exactly n_samples entries."""
    with patch("model.monte_carlo.runner.handle_intervention", return_value=FIXED_RESULT), \
         patch("model.monte_carlo.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        results = run_monte_carlo(
            base_input_dict=BASE_INPUT,
            soil_params=FakeSoilParams(),
            climate=FakeClimateData(),
            n_samples=5,
            create_forward_soil_model=fake_forward,
            create_inverse_soil_model=fake_inverse,
            n_cohorts=1,
            plot_index=0,
            seed=0,
        )
    assert len(results) == 5


def test_run_monte_carlo_deterministic_no_distributions():
    """With no distributions and zero soil/climate uncertainty, every call to
    handle_intervention receives identical inputs."""
    captured_inputs = []

    def capture(**kwargs):
        captured_inputs.append(kwargs["intervention_input"])
        return FIXED_RESULT

    with patch("model.monte_carlo.runner.handle_intervention", side_effect=capture), \
         patch("model.monte_carlo.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        run_monte_carlo(
            base_input_dict=BASE_INPUT,
            soil_params=FakeSoilParams(),
            climate=FakeClimateData(),
            n_samples=4,
            create_forward_soil_model=fake_forward,
            create_inverse_soil_model=fake_inverse,
            n_cohorts=1,
            plot_index=0,
            seed=42,
        )

    assert len(captured_inputs) == 4
    first = captured_inputs[0]
    for subsequent in captured_inputs[1:]:
        assert set(first.keys()) == set(subsequent.keys())
        for key in first:
            np.testing.assert_array_equal(
                np.asarray(first[key]),
                np.asarray(subsequent[key]),
                err_msg=f"Input '{key}' differs between samples despite zero uncertainty",
            )


def test_run_monte_carlo_uses_soil_override():
    """handle_intervention is called with soil_override set (not None) on every call."""
    with patch("model.monte_carlo.runner.handle_intervention", return_value=FIXED_RESULT) as mock_handle, \
         patch("model.monte_carlo.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        run_monte_carlo(
            base_input_dict=BASE_INPUT,
            soil_params=FakeSoilParams(),
            climate=FakeClimateData(),
            n_samples=3,
            create_forward_soil_model=fake_forward,
            create_inverse_soil_model=fake_inverse,
            n_cohorts=1,
            plot_index=0,
            seed=0,
        )

    assert mock_handle.call_count == 3
    for call in mock_handle.call_args_list:
        soil_override = call.kwargs.get("soil_override")
        assert soil_override is not None, "soil_override should be set on every call"


# ---------------------------------------------------------------------------
# Helpers for summarise_mc_results / write_mc_summary_csv
# ---------------------------------------------------------------------------

def _make_fake_result(n_years, offset=0.0):
    """Return a minimal object satisfying all field accesses in the getter dicts."""
    zeros = np.zeros(n_years)
    ones = np.ones(n_years)
    r = MagicMock()
    # base
    r.soil_base_emissions = zeros
    r.tree_base_emissions = zeros
    r.fire_base_emissions = zeros
    r.litter_base_emissions = zeros
    r.fertiliser_base_emissions = zeros
    r.crop_base_emissions = zeros
    r.emit_base_emissions = zeros
    # project
    r.soil_project_emissions = ones + offset
    r.tree_project_emissions = ones + offset
    r.fire_project_emissions = ones + offset
    r.litter_project_emissions = ones + offset
    r.fertiliser_project_emissions = ones + offset
    r.crop_project_emissions = ones + offset
    r.emit_project_emissions = ones + offset
    # diff
    r.soil_difference = ones + offset
    r.tree_difference = ones + offset
    r.fire_difference = ones + offset
    r.litter_difference = ones + offset
    r.fertiliser_difference = ones + offset
    r.crop_difference = ones + offset
    return r


# ---------------------------------------------------------------------------
# summarise_mc_results — structure
# ---------------------------------------------------------------------------

def test_summarise_mc_results_column_names():
    """Expected columns are present in every scenario dict."""
    results = [_make_fake_result(5) for _ in range(3)]
    summaries = summarise_mc_results(results)
    pools = ["soil", "tree", "fire", "litter", "fertiliser", "crop", "emit"]
    stats = ["mean", "std", "q05", "q25", "q50", "q75", "q95"]
    for scenario in (summaries.base, summaries.project, summaries.diff):
        for pool in pools:
            for stat in stats:
                assert f"{pool}_{stat}" in scenario, f"Missing column {pool}_{stat}"


def test_summarise_mc_results_all_scenarios_same_keys():
    """base, project, and diff dicts have identical column sets."""
    results = [_make_fake_result(5) for _ in range(3)]
    summaries = summarise_mc_results(results)
    assert set(summaries.base.keys()) == set(summaries.project.keys()) == set(summaries.diff.keys())


def test_summarise_mc_results_array_length():
    """Each column array has length n_years in every scenario dict."""
    n_years = 7
    results = [_make_fake_result(n_years) for _ in range(4)]
    summaries = summarise_mc_results(results)
    for scenario in (summaries.base, summaries.project, summaries.diff):
        for col, arr in scenario.items():
            assert len(arr) == n_years, f"Column {col} has length {len(arr)}, expected {n_years}"


# ---------------------------------------------------------------------------
# summarise_mc_results — values
# ---------------------------------------------------------------------------

def test_summarise_mc_results_zero_std_when_identical():
    """When all samples are identical, std is zero in every scenario."""
    results = [_make_fake_result(4, offset=1.0) for _ in range(10)]
    summaries = summarise_mc_results(results)
    pools = ["soil", "tree", "fire", "litter", "fertiliser", "crop", "emit"]
    for scenario in (summaries.base, summaries.project, summaries.diff):
        for pool in pools:
            np.testing.assert_array_equal(
                scenario[f"{pool}_std"],
                np.zeros(4),
                err_msg=f"std should be 0 for {pool} when all samples are identical",
            )


def test_summarise_mc_results_known_mean():
    """Mean of emit diff equals the known per-year mean across samples."""
    n_years = 3
    offsets = [1.0, 2.0, 4.0]
    results = [_make_fake_result(n_years, offset=o) for o in offsets]
    summaries = summarise_mc_results(results)
    # emit_difference per sample = 1 + offset; mean = 1 + mean(offsets)
    expected_mean = 1.0 + np.mean(offsets)
    np.testing.assert_allclose(
        summaries.diff["emit_mean"],
        np.full(n_years, expected_mean),
        rtol=1e-10,
    )


def test_summarise_mc_results_q50_is_median():
    """q50 of diff matches the expected median for a known set of samples."""
    n_years = 2
    offsets = [0.0, 2.0, 10.0]
    results = [_make_fake_result(n_years, offset=o) for o in offsets]
    summaries = summarise_mc_results(results)
    # emit_difference per sample = 1 + offset; median = 1 + 2.0 = 3.0
    np.testing.assert_allclose(summaries.diff["emit_q50"], np.full(n_years, 3.0), rtol=1e-10)


def test_summarise_mc_results_base_is_zero():
    """Baseline emissions are zero in _make_fake_result; base summary mean should be zero."""
    results = [_make_fake_result(3) for _ in range(5)]
    summaries = summarise_mc_results(results)
    np.testing.assert_array_equal(summaries.base["emit_mean"], np.zeros(3))


# ---------------------------------------------------------------------------
# write_mc_summary_csv
# ---------------------------------------------------------------------------

def test_write_mc_summary_csv_headers(tmp_path):
    """CSV header row contains 'year' followed by all summary column names."""
    results = [_make_fake_result(3) for _ in range(2)]
    scenario = summarise_mc_results(results).diff
    out = str(tmp_path / "mc_diff.csv")
    write_mc_summary_csv(scenario, out)

    with open(out, newline="") as f:
        header = next(csv.reader(f))

    assert header[0] == "year"
    assert set(header[1:]) == set(scenario.keys())


def test_write_mc_summary_csv_row_count(tmp_path):
    """CSV has exactly n_years data rows (plus one header row)."""
    n_years = 5
    results = [_make_fake_result(n_years) for _ in range(2)]
    scenario = summarise_mc_results(results).diff
    out = str(tmp_path / "mc_diff.csv")
    write_mc_summary_csv(scenario, out)

    with open(out, newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == n_years + 1  # header + data


def test_write_mc_summary_csv_year_column(tmp_path):
    """Year column contains 1, 2, ..., n_years."""
    n_years = 4
    results = [_make_fake_result(n_years) for _ in range(2)]
    scenario = summarise_mc_results(results).diff
    out = str(tmp_path / "mc_diff.csv")
    write_mc_summary_csv(scenario, out)

    with open(out, newline="") as f:
        years = [int(row["year"]) for row in csv.DictReader(f)]

    assert years == list(range(1, n_years + 1))


def test_write_mc_summary_csv_values_roundtrip(tmp_path):
    """Values read back from CSV match the summary dict to float precision."""
    n_years = 3
    results = [_make_fake_result(n_years, offset=float(i)) for i in range(5)]
    scenario = summarise_mc_results(results).diff
    out = str(tmp_path / "mc_diff.csv")
    write_mc_summary_csv(scenario, out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    col = "emit_mean"
    for year_idx, row in enumerate(rows):
        assert float(row[col]) == pytest.approx(float(scenario[col][year_idx]), rel=1e-9)
