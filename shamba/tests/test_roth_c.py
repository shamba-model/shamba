"""Tests for the RothC soil carbon model.

Covers:
  - get_rmf(): rate modifying factor (qualitative + code-logic tests)
  - forward_roth_c: shape, non-negativity, round-trip stability
  - inverse_roth_c: equilibrium pools sum to Ceq, non-negativity

Numeric regression tests (marked skip) must have their expected values confirmed
against the official RothC Python reference repository before activating.
See the FIXME comment in test_forward_solver_numeric_regression for the input spec.
"""

import types

import numpy as np
import pytest

from model.climate import ClimateData
import model.soil_params as soil_params
import model.soil_models.roth_c.roth_c as roth_c
import model.soil_models.roth_c.forward_roth_c as forward_roth_c
import model.soil_models.roth_c.inverse_roth_c as inverse_roth_c


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_soil(Cy0=40.0, clay=20.0):
    return soil_params.create({"Cy0": Cy0, "clay": clay})


def _make_wet_climate(temp=20.0):
    """Always-wet climate: rain > evap every month.

    When rain always exceeds evaporation the RMF code short-circuits and
    returns b.mean() = 1.0 without evaluating the temperature or cover factors.
    """
    return ClimateData(
        temperature=np.full(12, temp),
        rain=np.full(12, 100.0),
        evaporation=np.full(12, 40.0),
    )


def _make_drought_climate():
    """Severe drought: 4 months where rain is far below evap (deficit ~−60 mm/month)."""
    return ClimateData(
        temperature=np.full(12, 20.0),
        rain=np.array([100, 100, 100, 100, 100, 100, 20, 20, 20, 20, 100, 100], dtype=float),
        evaporation=np.array([40, 40, 40, 40, 40, 40, 80, 80, 80, 80, 40, 40], dtype=float),
    )


def _make_mild_drought_climate():
    """Mild drought: 4 months where rain is slightly below evap (deficit ~−10 mm/month)."""
    return ClimateData(
        temperature=np.full(12, 20.0),
        rain=np.array([100, 100, 100, 100, 100, 100, 60, 60, 60, 60, 100, 100], dtype=float),
        evaporation=np.array([40, 40, 40, 40, 40, 40, 70, 70, 70, 70, 40, 40], dtype=float),
    )


def _make_litter_input(n_years, carbon_per_year):
    """Create a mock litter object providing constant carbon input each year.

    forward_roth_c reads litter via emit.reduce_from_fire, which sums
    litter.output["above"]["carbon"] and litter.output["below"]["carbon"].
    """
    obj = types.SimpleNamespace()
    obj.output = {
        "above": {
            "carbon": np.full(n_years, carbon_per_year),
            "nitrogen": np.zeros(n_years),
            "DMon": np.zeros(n_years),
            "DMoff": np.zeros(n_years),
        },
        "below": {
            "carbon": np.zeros(n_years),
            "nitrogen": np.zeros(n_years),
            "DMon": np.zeros(n_years),
            "DMoff": np.zeros(n_years),
        },
    }
    return obj


# ---------------------------------------------------------------------------
# get_rmf tests
# ---------------------------------------------------------------------------

class TestGetRmf:

    def test_rmf_equals_one_when_rain_always_exceeds_evap(self):
        # When deficit = rain - evap > 0 every month, the code sets
        # rain_always_exceeds_evaporation = True and returns b.mean() = 1.0
        # before applying temperature or cover factors.
        soil = _make_soil()
        climate = _make_wet_climate()
        cover = np.zeros(12)
        n_years = 5
        result = roth_c.get_rmf(climate=climate, cover=cover, soil=soil, no_of_years=n_years)
        np.testing.assert_allclose(result, np.ones(n_years))

    def test_severe_drought_gives_lower_rmf_than_mild_drought(self):
        # Both climates go through the full RMF calculation (both have some months
        # where rain < evap). Severe drought accumulates a larger soil moisture deficit,
        # so the moisture factor b is lower → lower RMF.
        # Note: the "always wet" bypass returns b.mean()=1.0 *without* the temperature
        # factor, so wet vs drought is not a valid comparison for this test.
        soil = _make_soil()
        cover = np.zeros(12)
        rmf_mild = roth_c.get_rmf(_make_mild_drought_climate(), cover, soil, no_of_years=1)
        rmf_severe = roth_c.get_rmf(_make_drought_climate(), cover, soil, no_of_years=1)
        assert rmf_severe[0] < rmf_mild[0]

    def test_soil_cover_reduces_rmf_under_drought(self):
        # Covered soil uses a c factor of 0.6 instead of 1.0.
        soil = _make_soil()
        drought = _make_drought_climate()
        rmf_bare = roth_c.get_rmf(drought, np.zeros(12), soil, no_of_years=1)
        rmf_covered = roth_c.get_rmf(drought, np.ones(12), soil, no_of_years=1)
        assert rmf_covered[0] < rmf_bare[0]


# ---------------------------------------------------------------------------
# inverse_roth_c tests
# ---------------------------------------------------------------------------

class TestInverseRothC:

    def test_equilibrium_pools_sum_to_approximately_ceq(self):
        # The inverse solver searches for pools where sum(eq_C) + iom ≈ Ceq.
        soil = _make_soil(Cy0=40.0, clay=20.0)
        inv = inverse_roth_c.create(soil, _make_drought_climate(), cover=np.zeros(12))
        total_C = float(np.sum(inv.eq_C)) + soil.iom
        # The inverse solver uses a coarse 0.1-step grid search, so convergence
        # is approximate. 20% tolerance accounts for this grid resolution.
        assert abs(total_C - soil.Ceq) / soil.Ceq < 0.20

    def test_equilibrium_pools_are_non_negative(self):
        soil = _make_soil()
        inv = inverse_roth_c.create(soil, _make_drought_climate(), cover=np.zeros(12))
        assert all(v >= 0 for v in inv.eq_C)


# ---------------------------------------------------------------------------
# forward_roth_c tests
# ---------------------------------------------------------------------------

class TestForwardRothC:

    def test_soc_has_correct_shape(self):
        # fwd.SOC is a list of lists after marshmallow deserialisation.
        n_years = 5
        soil = _make_soil()
        Ci = np.array([0.1, 10.0, 0.0, 0.0])
        fwd = forward_roth_c.create(soil, _make_drought_climate(), np.zeros(12), Ci, n_years)
        assert np.array(fwd.SOC).shape == (n_years + 1, 4)

    def test_pools_do_not_go_substantially_negative(self):
        # The RothC ODE solver does not enforce strict positivity. Fast-decaying
        # pools (DPM k ≈ 10×RMF) can undershoot to tiny floating-point negatives
        # when approaching zero with no carbon input. The meaningful constraint is
        # that no pool goes substantially negative (i.e. > 1e-6 t C/ha below zero).
        # fwd.SOC is a list of lists after marshmallow deserialisation.
        Ci = np.array([0.1, 10.0, 0.0, 0.0])
        fwd = forward_roth_c.create(
            _make_soil(), _make_drought_climate(), np.zeros(12), Ci, no_of_years=10
        )
        assert np.all(np.array(fwd.SOC) >= -1e-6)

    def test_inverse_forward_round_trip_is_stable(self):
        # Get equilibrium pools from the inverse solver.
        # Run forward with those pools and the matching equilibrium carbon input.
        # Total carbon should remain approximately constant (within 30% over 10 years).
        #
        # Note: the inverse solver uses a coarse 0.1-step grid search so the
        # round-trip is only approximate. The 30% tolerance accounts for this.
        n_years = 10
        soil = _make_soil(Cy0=40.0, clay=20.0)
        climate = _make_drought_climate()
        cover = np.zeros(12)

        inv = inverse_roth_c.create(soil, climate, cover)
        Ci = np.array(inv.eq_C)
        litter = _make_litter_input(n_years, inv.input_C)

        fwd = forward_roth_c.create(
            soil, climate, cover, Ci, n_years,
            litter=[litter], fire=np.zeros(n_years),
        )

        total_start = float(np.sum(fwd.SOC[0]))
        total_end = float(np.sum(fwd.SOC[-1]))
        relative_change = abs(total_end - total_start) / total_start
        assert relative_change < 0.30, (
            f"Carbon pools should be approximately stable near equilibrium; "
            f"relative change was {relative_change:.1%}"
        )

    @pytest.mark.skip(
        reason=(
            "Numeric regression values must be confirmed against the official "
            "RothC Python reference repository before activating this test. "
            "See the FIXME comment below for the input specification."
        )
    )
    def test_forward_solver_numeric_regression(self):
        # FIXME: Run the official RothC Python reference repository on these inputs
        # and record the expected pool values (DPM, RPM, BIO, HUM) at year 5 below.
        #
        # Inputs to use:
        #   Cy0=40.0, clay=20.0, depth=30 (default)
        #   climate: temp=20°C constant, rain=[100]*6+[20]*6, evap=[40]*6+[80]*6 mm
        #   cover = zeros (bare soil), no_of_years=5
        #   Ci = [0.1, 10.0, 0.0, 0.0]
        #   litter input: use inv.input_C from inverse_roth_c.create() on same soil/climate
        #
        expected_SOC_year5 = [np.nan, np.nan, np.nan, np.nan]  # REPLACE with reference values

        n_years = 5
        soil = _make_soil(Cy0=40.0, clay=20.0)
        climate = _make_drought_climate()
        cover = np.zeros(12)
        Ci = np.array([0.1, 10.0, 0.0, 0.0])

        inv = inverse_roth_c.create(soil, climate, cover)
        litter = _make_litter_input(n_years, inv.input_C)

        fwd = forward_roth_c.create(
            soil, climate, cover, Ci, n_years,
            litter=[litter], fire=np.zeros(n_years),
        )
        np.testing.assert_allclose(fwd.SOC[n_years], expected_SOC_year5, rtol=1e-4)
