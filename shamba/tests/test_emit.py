"""Tests for individual functions in emit.py.

These tests exercise the building-block functions (reduce_from_fire, soc_sink,
tree_sink, nitrogen_emit, fert_emit) independently of the full model run.
"""

import types

import numpy as np
import pytest

import model.emit as Emit
from model.common.constants import (
    C_to_CO2_conversion_factor,
    combustion_factor,
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
