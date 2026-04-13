"""Tests for data validation functions in data_handler.py.

Per the transparency principle in CLAUDE.md: missing management columns must raise
a clear error. Explicit zeros are required; silent defaults are forbidden.
"""

import numpy as np
import pytest
from model.common.data_handler import (
    validate_required_mgmt_keys,
    validate_species_data,
    resolve_evap_pet,
    REQUIRED_MGMT_KEYS,
)


def _minimal_valid_mgmt_dict():
    return {key: np.zeros(5) for key in REQUIRED_MGMT_KEYS}


class TestValidateRequiredMgmtKeys:

    def test_passes_when_all_required_keys_present(self):
        assert validate_required_mgmt_keys(_minimal_valid_mgmt_dict()) == []

    def test_error_names_missing_keys_in_single_message(self):
        # All missing keys must be reported in one message (not one per key).
        data = _minimal_valid_mgmt_dict()
        del data["fire_on_base"]
        del data["base_sf_qty1"]
        errors = validate_required_mgmt_keys(data)
        assert len(errors) == 1
        assert "fire_on_base" in errors[0]
        assert "base_sf_qty1" in errors[0]


class TestValidateSpeciesData:

    def test_passes_when_age_and_diam_present_for_declared_species(self):
        data = {
            "base_species1": np.array([1.0]),
            "age_sp1": np.array([1.0, 2.0, 3.0]),
            "diam_sp1": np.array([5.0, 8.0, 12.0]),
        }
        assert validate_species_data(data) == []

    def test_passes_when_age_and_biomass_present_without_diam(self):
        data = {
            "base_species1": np.array([1.0]),
            "age_sp1": np.array([1.0, 2.0, 3.0]),
            "biomass_sp1": np.array([10.0, 25.0, 50.0]),
        }
        assert validate_species_data(data) == []

    def test_error_when_size_data_missing_for_declared_species(self):
        # base_species1 = 2 → age_sp2 and at least one of diam_sp2/biomass_sp2 required.
        data = {"base_species1": np.array([2.0])}
        errors = validate_species_data(data)
        assert any("age_sp2" in e for e in errors)
        assert any("diam_sp2" in e and "biomass_sp2" in e for e in errors)

    def test_no_false_positive_for_age_when_size_absent(self):
        # age_sp3 present but neither diam_sp3 nor biomass_sp3 absent —
        # must report the combined size error, but not report age_sp3 as missing.
        data = {
            "proj_species1": np.array([3.0]),
            "age_sp3": np.array([1.0, 2.0]),
        }
        errors = validate_species_data(data)
        assert any("diam_sp3" in e and "biomass_sp3" in e for e in errors)
        assert not any("age_sp3" in e for e in errors)

    def test_biomass_sp_recognised_as_valid_header(self):
        from model.common.data_handler import get_header_type
        assert get_header_type("biomass_sp1") == "non-negative float"
        assert get_header_type("biomass_sp3") == "non-negative float"


class TestResolveEvapPet:

    def test_evap_wins_and_pet_discarded_when_both_present(self):
        evap = np.array([10.0, 20.0])
        result = resolve_evap_pet({"evap": evap.copy(), "pet": np.array([99.0, 99.0])})
        assert "pet" not in result
        np.testing.assert_array_equal(result["evap"], evap)

    def test_pet_converted_to_evap_via_open_pan_factor(self):
        # evap = pet / 0.75 → pet=75 gives evap=100
        result = resolve_evap_pet({"pet": np.array([75.0, 150.0])})
        assert "pet" not in result
        np.testing.assert_allclose(result["evap"], [100.0, 200.0])

    def test_raises_when_neither_evap_nor_pet_present(self):
        with pytest.raises(ValueError):
            resolve_evap_pet({"rain": np.array([50.0])})
