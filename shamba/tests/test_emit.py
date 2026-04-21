"""Tests for individual functions in emit.py.

These tests exercise the building-block functions (reduce_from_fire, soc_sink,
tree_sink, nitrogen_emit, fert_emit) independently of the full model run.
"""

import types

import numpy as np
import pytest

import model.emit as Emit
from model.emit import EmissionFactors
from model.common.constants import (
    C_to_CO2_conversion_factor,
    combustion_factor_default,
    ef_burn_default,
    ef_N_inputs_default,
    volatile_frac_organic_fertiliser_default,
    volatile_frac_synthetic_fertiliser_default,
    GWP_list,
    DEFAULT_GWP,
)

GWP = GWP_list[DEFAULT_GWP]


def _make_biomass_model(n_years, above_carbon, below_carbon):
    """Create a minimal mock biomass object with the .output structure that
    reduce_from_fire, nitrogen_emit, and fert_emit read from."""
    obj = types.SimpleNamespace()
    obj.output = {
        "above": {
            "carbon": np.array(above_carbon, dtype=float),
            "nitrogen": np.zeros(n_years),
            "DMon": np.zeros(n_years),
            "DMoff": np.zeros(n_years),
        },
        "below": {
            "carbon": np.array(below_carbon, dtype=float),
            "nitrogen": np.zeros(n_years),
            "DMon": np.zeros(n_years),
            "DMoff": np.zeros(n_years),
        },
    }
    return obj


class TestReduceFromFire:

    def test_no_fire_returns_unmodified_above_plus_below(self):
        # fire=0 every year → output equals sum of above + below carbon inputs.
        n = 3
        crop = [_make_biomass_model(n, [10.0, 10.0, 10.0], [5.0, 5.0, 5.0])]
        fire = np.zeros(n)
        crop_out, _ = Emit.reduce_from_fire(n, crop=crop, fire=fire)
        np.testing.assert_allclose(crop_out, [15.0, 15.0, 15.0])

    def test_fire_reduces_crop_above_ground_by_combustion_factor(self):
        # combustion_factor["crop"] = 0.85
        # After fire: above = above * (1 - 0.85) = above * 0.15
        n = 3
        crop = [_make_biomass_model(n, [10.0, 10.0, 10.0], [5.0, 5.0, 5.0])]
        fire = np.array([1.0, 0.0, 0.0])  # fire in year 0 only
        crop_out, _ = Emit.reduce_from_fire(n, crop=crop, fire=fire)
        # Year 0: above=10*(1-0.85)=1.5, below=5 → total=6.5
        # Years 1-2: above=10, below=5 → total=15
        np.testing.assert_allclose(crop_out, [6.5, 15.0, 15.0])

    def test_fire_reduces_tree_above_ground_by_tree_combustion_factor(self):
        # combustion_factor["tree"] = 0.74
        # After fire: above = above * (1 - 0.74) = above * 0.26
        n = 2
        tree = [_make_biomass_model(n, [10.0, 10.0], [5.0, 5.0])]
        fire = np.array([1.0, 0.0])
        _, tree_out = Emit.reduce_from_fire(n, tree=tree, fire=fire)
        # Year 0: above=10*(1-0.74)=2.6, below=5 → total=7.6
        # Year 1: above=10, below=5 → total=15
        np.testing.assert_allclose(tree_out, [7.6, 15.0])

    def test_empty_inputs_return_zeros(self):
        n = 5
        crop_out, tree_out = Emit.reduce_from_fire(n, crop=[], tree=[], litter=[], fire=np.zeros(n))
        np.testing.assert_array_equal(crop_out, np.zeros(n))
        np.testing.assert_array_equal(tree_out, np.zeros(n))


class TestSocSink:

    def test_numeric_value_1_tC_increase(self):
        # A 1 t C/ha increase per year → delta = 1 * 44/12 = 3.6667 t CO2/ha/yr
        # C_to_CO2_conversion_factor = 44/12
        mock_fwd = types.SimpleNamespace()
        mock_fwd.SOC = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        result = Emit.soc_sink(mock_fwd, no_of_years=1)
        np.testing.assert_allclose(result, [44.0 / 12], rtol=1e-6)

    def test_constant_soc_returns_zero(self):
        mock_fwd = types.SimpleNamespace()
        mock_fwd.SOC = np.full((3, 4), [10.0, 0.0, 0.0, 0.0])
        result = Emit.soc_sink(mock_fwd, no_of_years=2)
        np.testing.assert_allclose(result, [0.0, 0.0])


class TestTreeSink:

    def test_numeric_value(self):
        # Total biomass increases by 2 per year (from 8 to 10 to 12).
        # delta = 2 * 44/12 per year
        mock_tree = types.SimpleNamespace()
        mock_tree.stand_biomass = np.array([[5.0, 3.0], [6.0, 4.0], [7.0, 5.0]])
        result = Emit.tree_sink([mock_tree], no_of_years=2)
        expected = np.array([2 * 44.0 / 12, 2 * 44.0 / 12])
        np.testing.assert_allclose(result, expected)

    def test_multiple_trees_are_summed(self):
        # Two trees each contributing 2 units/year → total delta = 4 * 44/12
        mock_tree = types.SimpleNamespace()
        mock_tree.stand_biomass = np.array([[5.0, 3.0], [6.0, 4.0]])
        result = Emit.tree_sink([mock_tree, mock_tree], no_of_years=1)
        expected = np.array([4 * 44.0 / 12])
        np.testing.assert_allclose(result, expected)


class TestZeroInputsGiveZeroEmissions:

    def test_fert_emit_zero_for_empty_inputs(self):
        result = Emit.fert_emit(litter=[], fert=[], no_of_years=5, gwp=GWP)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_nitrogen_emit_zero_for_empty_inputs(self):
        result = Emit.nitrogen_emit(
            no_of_years=5,
            crop=[],
            tree=[],
            litter=[],
            fire=np.zeros(5),
            gwp=GWP,
        )
        np.testing.assert_array_equal(result, np.zeros(5))


# ---------------------------------------------------------------------------
# EmissionFactors injection — verify custom values flow through each helper
# ---------------------------------------------------------------------------

def _make_nitrogen_model(n_years, nitrogen):
    """Minimal biomass object with only the nitrogen output populated."""
    obj = types.SimpleNamespace()
    obj.output = {
        "above": {
            "carbon": np.zeros(n_years),
            "nitrogen": np.array(nitrogen, dtype=float),
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


def _make_dm_model(n_years, DMon, DMoff=None):
    """Minimal biomass object with only the DMon/DMoff outputs populated (for fire_emit)."""
    obj = types.SimpleNamespace()
    obj.output = {
        "above": {
            "carbon": np.zeros(n_years),
            "nitrogen": np.zeros(n_years),
            "DMon": np.array(DMon, dtype=float),
            "DMoff": np.zeros(n_years) if DMoff is None else np.array(DMoff, dtype=float),
        },
        "below": {
            "carbon": np.zeros(n_years),
            "nitrogen": np.zeros(n_years),
            "DMon": np.zeros(n_years),
            "DMoff": np.zeros(n_years),
        },
    }
    return obj


class TestEmissionFactorsInjection:
    """Verify that each helper in emit.py uses the emission_factors it is given,
    not hardcoded defaults.  Each test compares a default run to a run with an
    explicitly altered factor and asserts the results differ."""

    def test_fert_emit_organic_volatile_frac(self):
        """Doubling volatile_frac_organic_fertiliser changes fert_emit output."""
        n = 3
        litter = [_make_nitrogen_model(n, [1.0, 1.0, 1.0])]
        default_result = Emit.fert_emit(litter=litter, fert=[], no_of_years=n, gwp=GWP)
        custom_result = Emit.fert_emit(
            litter=litter, fert=[], no_of_years=n, gwp=GWP,
            volatile_frac_organic_fertiliser=volatile_frac_organic_fertiliser_default * 2,
        )
        assert not np.allclose(default_result, custom_result), (
            "fert_emit should use the supplied volatile_frac_organic_fertiliser"
        )
        # TODO: replace "not allclose" with exact expected values. With N=[1,1,1] and
        # volatile_frac_organic=0.21 (default):
        #   emit_per_year = 1 * (1 - 0.21) * ef_N_inputs * N_to_N2O_conv * gwp["N2O"]
        #                 = 0.79 * 0.01 * (44/28) * 273
        # Assert default_result ≈ [emit_per_year, ...] and custom_result ≈ [custom_emit, ...]
        # where custom_emit uses volatile_frac=0.42.

    def test_fert_emit_synthetic_volatile_frac(self):
        """Doubling volatile_frac_synthetic_fertiliser changes fert_emit output."""
        n = 3
        fert = [_make_nitrogen_model(n, [1.0, 1.0, 1.0])]
        default_result = Emit.fert_emit(litter=[], fert=fert, no_of_years=n, gwp=GWP)
        custom_result = Emit.fert_emit(
            litter=[], fert=fert, no_of_years=n, gwp=GWP,
            volatile_frac_synthetic_fertiliser=volatile_frac_synthetic_fertiliser_default * 2,
        )
        assert not np.allclose(default_result, custom_result), (
            "fert_emit should use the supplied volatile_frac_synthetic_fertiliser"
        )
        # TODO: replace "not allclose" with exact expected values. With N=[1,1,1] and
        # volatile_frac_synthetic=0.11 (default):
        #   emit_per_year = 1 * (1 - 0.11) * ef_N_inputs * N_to_N2O_conv * gwp["N2O"]
        #                 = 0.89 * 0.01 * (44/28) * 273
        # Assert default_result ≈ [emit_per_year, ...] and custom_result uses 0.22.

    def test_nitrogen_emit_ef_N_inputs(self):
        """Doubling ef_N_inputs doubles nitrogen_emit output."""
        n = 3
        crop = [_make_nitrogen_model(n, [2.0, 2.0, 2.0])]
        fire = np.zeros(n)
        default_result = Emit.nitrogen_emit(
            no_of_years=n, crop=crop, tree=[], litter=[], fire=fire, gwp=GWP,
        )
        doubled_result = Emit.nitrogen_emit(
            no_of_years=n, crop=crop, tree=[], litter=[], fire=fire, gwp=GWP,
            ef_N_inputs=ef_N_inputs_default * 2,
        )
        np.testing.assert_allclose(doubled_result, default_result * 2, rtol=1e-6)

    def test_fire_emit_ef_burn(self):
        """Changing ef_burn changes fire_emit output when fire is present."""
        n = 3
        crop = [_make_dm_model(n, DMon=[10.0, 10.0, 10.0])]
        fire = np.ones(n)
        default_result = Emit.fire_emit(
            crop=crop, tree=[], litter=[], fire=fire, no_of_years=n, gwp=GWP,
        )
        custom_ef = {**ef_burn_default, "crop_N2O": ef_burn_default["crop_N2O"] * 2}
        custom_result = Emit.fire_emit(
            crop=crop, tree=[], litter=[], fire=fire, no_of_years=n, gwp=GWP,
            ef_burn=custom_ef,
        )
        assert not np.allclose(default_result, custom_result), (
            "fire_emit should use the supplied ef_burn"
        )
        # TODO: replace "not allclose" with exact expected values. With DMon=[10,10,10],
        # fire=ones, combustion_factor["crop"]=0.85, DMoff=zeros (burn_off path is zero):
        #   crop_CO2_ef = ef_burn["crop_CH4"] * gwp["CH4"] + ef_burn["crop_N2O"] * gwp["N2O"]
        #               = 2.7 * 27 + 0.07 * 273 = 72.9 + 19.11 = 92.01
        #   emit_per_year = 10 * 1 * 0.85 * 92.01 * 0.001 = 0.782085
        # Assert default_result ≈ [0.782085, 0.782085, 0.782085].
        # Assert custom_result uses doubled crop_N2O=0.14 and calculate accordingly.

    def test_fire_emit_combustion_factor(self):
        """Changing combustion_factor changes fire_emit output when fire is present."""
        n = 3
        crop = [_make_dm_model(n, DMon=[10.0, 10.0, 10.0])]
        fire = np.ones(n)
        default_result = Emit.fire_emit(
            crop=crop, tree=[], litter=[], fire=fire, no_of_years=n, gwp=GWP,
        )
        custom_cf = {**combustion_factor_default, "crop": combustion_factor_default["crop"] * 0.5}
        custom_result = Emit.fire_emit(
            crop=crop, tree=[], litter=[], fire=fire, no_of_years=n, gwp=GWP,
            combustion_factor=custom_cf,
        )
        assert not np.allclose(default_result, custom_result), (
            "fire_emit should use the supplied combustion_factor"
        )
        # TODO: replace "not allclose" with exact expected values. combustion_factor is
        # linear in fire_emit, so halving crop combustion_factor (0.85 → 0.425) should
        # halve the on-farm fire emission term. Assert custom_result ≈ default_result * 0.5.

    def test_emit_create_with_custom_emission_factors(self):
        """emit.create() with a custom EmissionFactors produces a different result
        than the default.  Uses fert inputs so the volatile_frac path is exercised."""
        n = 3
        fert = [_make_nitrogen_model(n, [1.0, 1.0, 1.0])]
        default_result = Emit.create(no_of_years=n, fert=fert, gwp=GWP)
        custom_ef = EmissionFactors(
            volatile_frac_synthetic_fertiliser=volatile_frac_synthetic_fertiliser_default * 3,
        )
        custom_result = Emit.create(no_of_years=n, fert=fert, gwp=GWP, emission_factors=custom_ef)
        assert not np.allclose(default_result, custom_result), (
            "emit.create() should pass emission_factors through to fert_emit"
        )
        # TODO: replace "not allclose" with exact expected values. This is an integration
        # check that emission_factors routes through create() to fert_emit. With fert N=[1,1,1]
        # and volatile_frac_synthetic=0.11 (default), the expected per-year value is:
        #   1 * (1 - 0.11) * ef_N_inputs * N_to_N2O_conv * gwp["N2O"]
        #   = 0.89 * 0.01 * (44/28) * 273
        # With tripled volatile_frac=0.33, compute the analogous value and assert exactly.
