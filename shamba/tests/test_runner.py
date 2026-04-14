"""Tests for monte_carlo/runner.py (Issue 3).

Covers:
- run_monte_carlo returns a list of length n_samples
- When no distribution_dict and zero climate/soil uncertainty, all results are identical
- handle_intervention is called with soil_override set (not None) on every call
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from model.monte_carlo.runner import run_monte_carlo


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
    "Temp": np.array([20.0] * 12),
    "Rain": np.array([80.0] * 12),
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
