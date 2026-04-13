"""Tests for allometric equations and growth curve functions in tree_growth.py.

Expected values are computed by hand from the published formulae:
  - Ryan (2010): ln(AGB) = 2.601*ln(DBH) - 3.629
  - Chave et al. (2005): ln(AGB) = polyval(params, ln(DBH)), then * wood_density * carbon
  - Tumwebaze et al. (2013): sum of two log-allometric terms * carbon
  - Growth curve functions: directly from model description equations 6.1–6.4
"""

import math
import types

import numpy as np
import pytest

import model.tree_growth as TreeGrowth


def _make_tree_params(wood_dens=0.6, carbon=0.48):
    """Return a minimal mock object with the attributes allometric functions use."""
    return types.SimpleNamespace(wood_dens=wood_dens, carbon=carbon)


class TestRyanAllometric:
    """C. Ryan, biotropica (2010): AGB = exp(2.601*ln(DBH) - 3.629), units kg C"""

    def test_known_value_at_dbh_10(self):
        # Hand calculation:
        #   ln(10) = 2.302585
        #   log_biomass = 2.601 * 2.302585 - 3.629 = 5.989024 - 3.629 = 2.360024
        #   AGB = exp(2.360024) ≈ 10.591 kg C
        tp = _make_tree_params()
        result = TreeGrowth.ryan(dbh=10.0, tree_params=tp)
        expected = math.exp(2.601 * math.log(10.0) - 3.629)
        assert result == pytest.approx(expected, rel=1e-6)
        assert result == pytest.approx(10.591, rel=1e-3)

    def test_zero_dbh_returns_zero(self):
        assert TreeGrowth.ryan(dbh=0.0, tree_params=_make_tree_params()) == 0.0


class TestChaveDryAllometric:
    """Chave et al. (2005) dry forest.

    Formula: AGB = exp(polyval([-0.0281, 0.207, 1.784, -0.730], ln(DBH))) * WD * carbon
    """

    def test_known_value_at_dbh_10_wd_0p6_carbon_0p48(self):
        # Hand calculation:
        #   ln(10) = 2.302585
        #   polyval([-0.0281, 0.207, 1.784, -0.730], 2.302585)
        #     = -0.0281*(2.302585)^3 + 0.207*(2.302585)^2 + 1.784*(2.302585) - 0.730
        #     ≈ -0.342947 + 1.097493 + 4.107832 - 0.730000 = 4.132378
        #   raw_agb = exp(4.132378) ≈ 62.326
        #   AGB = 62.326 * 0.6 * 0.48 ≈ 17.950 kg C
        tp = _make_tree_params(wood_dens=0.6, carbon=0.48)
        result = TreeGrowth.chave_dry(dbh=10.0, tree_params=tp)
        log10 = math.log(10.0)
        log_raw_agb = np.polyval([-0.0281, 0.207, 1.784, -0.730], log10)
        expected = math.exp(log_raw_agb) * 0.6 * 0.48
        assert result == pytest.approx(expected, rel=1e-6)
        assert result == pytest.approx(17.950, rel=1e-2)

    def test_higher_wood_density_gives_higher_agb(self):
        assert (
            TreeGrowth.chave_dry(10.0, _make_tree_params(wood_dens=0.8))
            > TreeGrowth.chave_dry(10.0, _make_tree_params(wood_dens=0.4))
        )


class TestTumwebazeMarkhamiaAllometric:
    """Tumwebaze et al. (2013) Markhamia.

    Formula: AGB = (exp(polyval([2.63, -4.91], ln(DBH)))
                  + exp(polyval([2.43, -3.08], ln(DBH)))) * carbon
    """

    def test_known_value_at_dbh_10_carbon_0p48(self):
        # Hand calculation:
        #   ln(10) = 2.302585
        #   term1 = exp(2.63*2.302585 - 4.91) = exp(6.055799 - 4.91) = exp(1.145799) ≈ 3.145
        #   term2 = exp(2.43*2.302585 - 3.08) = exp(5.595282 - 3.08) = exp(2.515282) ≈ 12.370
        #   AGB = (3.145 + 12.370) * 0.48 ≈ 7.447 kg C
        tp = _make_tree_params(carbon=0.48)
        result = TreeGrowth.tumwebaze_markhamia(dbh=10.0, tree_params=tp)
        log10 = math.log(10.0)
        term1 = math.exp(np.polyval([2.63, -4.91], log10))
        term2 = math.exp(np.polyval([2.43, -3.08], log10))
        expected = (term1 + term2) * 0.48
        assert result == pytest.approx(expected, rel=1e-6)
        assert result == pytest.approx(7.447, rel=1e-2)


class TestGrowthCurveFunctions:
    """Tests for the standalone growth curve functions (Eqs. 6.1–6.4 in model description)."""

    def test_linear_function(self):
        # Eq. 6.1: f(x) = a*x
        assert TreeGrowth.linear_function(5.0, 2.0) == pytest.approx(10.0)
        assert TreeGrowth.linear_function(0.0, 2.0) == 0.0

    def test_exponential_1param_function(self):
        # Eq. 6.2: f(x) = (1+a)^x - 1
        # At x=0: f=0; at x=1, a=1: f=(1+1)^1 - 1 = 1
        assert TreeGrowth.exponential_1param_function(0.0, 0.5) == pytest.approx(0.0)
        assert TreeGrowth.exponential_1param_function(1.0, 1.0) == pytest.approx(1.0)

    def test_hyperbolic_function(self):
        # Eq. 6.3: f(x) = a*(1 - exp(-b*x))
        # At x→∞: f→a; at x=0: f=0
        a, b = 100.0, 0.5
        assert TreeGrowth.hyperbolic_function(0.0, a, b) == pytest.approx(0.0)
        # At large x, approaches asymptote a
        assert TreeGrowth.hyperbolic_function(100.0, a, b) == pytest.approx(a, rel=1e-5)

    def test_logistic_function(self):
        # Eq. 6.4: f(x) = a / (1 + exp(-b*(x-c)))
        # At x=c: f = a/2 (inflection point)
        a, b, c = 100.0, 0.5, 10.0
        assert TreeGrowth.logistic_function(c, a, b, c) == pytest.approx(a / 2.0)

    def test_fit_produces_monotonically_increasing_hyperbolic_curve(self):
        # On monotonically increasing training data, the fitted hyperbolic curve
        # should also be monotonically increasing.
        age = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        biomass = np.array([0.5, 1.2, 4.0, 10.0, 22.0])
        all_fit_data, all_fit_params, _ = TreeGrowth.fit(age, biomass)
        ages_fine = np.linspace(1, 20, 100)
        curve = TreeGrowth.hyperbolic_function(ages_fine, *all_fit_params["hyp"])
        assert all(np.diff(curve) > 0)


class TestFromCsvBiomassInput:
    """Tests that from_csv uses directly provided biomass data when available,
    bypassing the allometric equation."""

    def _make_tree_params(self):
        return types.SimpleNamespace(wood_dens=0.6, carbon=0.48, root_to_shoot=0.26, nitrogen=0.01)

    def _base_input(self):
        return {
            "age_sp1": np.array([1.0, 2.0, 5.0, 10.0]),
            "diam_sp1": np.array([5.0, 8.0, 12.0, 18.0]),
        }

    def test_uses_provided_biomass_directly(self):
        # Biomass values chosen to be inconsistent with any allometric equation —
        # if the model computes from diameter instead, this test will fail.
        biomass = np.array([10.0, 25.0, 80.0, 200.0])
        input_data = {**self._base_input(), "biomass_sp1": biomass}

        growth = TreeGrowth.from_csv(self._make_tree_params(), "chave dry", input_data, sp_index=1)

        np.testing.assert_array_equal(growth.biomass, biomass)

    def test_falls_back_to_allometry_when_biomass_not_provided(self):
        input_data = self._base_input()
        tp = self._make_tree_params()

        growth = TreeGrowth.from_csv(tp, "chave dry", input_data, sp_index=1)

        expected = TreeGrowth.get_biomass(input_data["diam_sp1"], "chave dry", tp)
        np.testing.assert_array_almost_equal(growth.biomass, expected)

    def test_provided_biomass_independent_of_allometric_key(self):
        # When biomass is supplied directly, changing the allometric key must not
        # change the biomass used for fitting.
        biomass = np.array([10.0, 25.0, 80.0, 200.0])
        input_data = {**self._base_input(), "biomass_sp1": biomass}
        tp = self._make_tree_params()

        growth_dry = TreeGrowth.from_csv(tp, "chave dry", input_data, sp_index=1)
        growth_ryan = TreeGrowth.from_csv(tp, "ryan", input_data, sp_index=1)

        np.testing.assert_array_equal(growth_dry.biomass, growth_ryan.biomass)

    def test_age_and_biomass_only_no_diam(self):
        # diam_sp1 absent: valid when biomass_sp1 is present.
        age = np.array([1.0, 2.0, 5.0, 10.0])
        biomass = np.array([10.0, 25.0, 80.0, 200.0])
        input_data = {"age_sp1": age, "biomass_sp1": biomass}

        growth = TreeGrowth.from_csv(self._make_tree_params(), "chave dry", input_data, sp_index=1)

        np.testing.assert_array_equal(growth.biomass, biomass)
        assert len(growth.tree_diameter) == 0
