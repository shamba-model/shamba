"""Tests for monte_carlo/distribution_handler.py (Issue 1).

Covers:
- Valid CSV with all seven distribution types loads without errors
- Unknown parameter name → plain-language error
- Unsupported distribution type → error
- Negative spread → error
- Negative min_abs → error
- min_abs on skew_normal / beta → error
- Asymmetric spread on a symmetric distribution → error
- beta on a non-[0,1] parameter → error
- Multiple errors reported together (not fail-fast)
- Warnings emitted for edge cases (base=0, min_abs dominates, fraction breach, etc.)
- Valid rows with no errors produce correct DistributionSpec entries
"""
import warnings
import pytest
import numpy as np

from model.monte_carlo.distribution_handler import (
    DistributionSpec,
    load_distributions,
)


# ---------------------------------------------------------------------------
# Shared base input dict used across most tests
# ---------------------------------------------------------------------------

BASE_DICT = {
    "crop_proj_yd1":  np.array([5.0, 6.0, 7.0]),   # vector, non-fraction
    "base_lit_qty1":  np.array([2.0, 2.0, 2.0]),   # vector, non-fraction
    "base_sf_qty1":   np.array([1.0, 1.0, 1.0]),   # vector, non-fraction
    "rain":           np.array([80.0] * 12),        # vector, non-fraction
    "temp":           np.array([20.0] * 12),        # vector, non-fraction (may be near 0)
    "crop_base_left1": np.array([0.3, 0.3, 0.3]),  # fraction parameter [0, 1]
    "thin_proj_cohort1": np.array([0.0, 0.2, 0.0]), # fraction parameter
}


def write_csv(tmp_path, content: str, filename="distributions.csv") -> str:
    p = tmp_path / filename
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# Valid load — all seven distributions
# ---------------------------------------------------------------------------

def test_load_all_seven_distributions(tmp_path):
    """A CSV with one row per distribution type loads without errors."""
    csv = write_csv(tmp_path, (
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_proj_yd1,normal,0.3,0.3,\n"
        "base_lit_qty1,truncated_normal,0.4,0.4,\n"
        "base_sf_qty1,lognormal,0.35,0.35,\n"
        "rain,uniform,0.2,0.3,\n"
        "base_lit_qty1,triangular,0.15,0.25,\n"
        "crop_proj_yd1,skew_normal,0.2,0.5,\n"
        "crop_base_left1,beta,2.0,5.0,\n"
    ))
    # beta is on a fraction parameter — valid
    # last two rows reuse parameters, which is allowed (last one wins per param)
    specs = load_distributions(csv, BASE_DICT)
    assert "crop_proj_yd1" in specs
    assert "crop_base_left1" in specs
    assert specs["rain"].distribution == "uniform"


def test_load_returns_correct_spec_values(tmp_path):
    """Parsed DistributionSpec contains the values from the CSV row."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_proj_yd1,normal,0.30,0.30,0.5\n"
    )
    specs = load_distributions(csv, BASE_DICT)
    spec = specs["crop_proj_yd1"]
    assert isinstance(spec, DistributionSpec)
    assert spec.distribution == "normal"
    assert spec.spread_lower == pytest.approx(0.30)
    assert spec.spread_upper == pytest.approx(0.30)
    assert spec.min_abs == pytest.approx(0.5)


def test_load_min_abs_absent_is_none(tmp_path):
    """When min_abs column is empty, spec.min_abs is None."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_proj_yd1,normal,0.30,0.30,\n"
    )
    specs = load_distributions(csv, BASE_DICT)
    assert specs["crop_proj_yd1"].min_abs is None


def test_load_min_abs_column_absent(tmp_path):
    """CSV without a min_abs column is valid; spec.min_abs is None."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,normal,0.30,0.30\n"
    )
    specs = load_distributions(csv, BASE_DICT)
    assert specs["crop_proj_yd1"].min_abs is None


def test_empty_spread_upper_copies_from_spread_lower(tmp_path):
    """Empty spread_upper is treated as equal to spread_lower (convenience for symmetric distributions)."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,normal,0.30,\n"
    )
    specs = load_distributions(csv, BASE_DICT)
    spec = specs["crop_proj_yd1"]
    assert spec.spread_upper == pytest.approx(spec.spread_lower)


# ---------------------------------------------------------------------------
# Error: unknown parameter
# ---------------------------------------------------------------------------

def test_unknown_parameter_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "not_a_real_param,normal,0.3,0.3\n"
    )
    with pytest.raises(ValueError, match="not_a_real_param"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: unsupported distribution
# ---------------------------------------------------------------------------

def test_unsupported_distribution_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,pareto,0.3,0.3\n"
    )
    with pytest.raises(ValueError, match="pareto"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: negative / zero spread
# ---------------------------------------------------------------------------

def test_zero_spread_lower_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,normal,0.0,0.3\n"
    )
    with pytest.raises(ValueError, match="spread_lower"):
        load_distributions(csv, BASE_DICT)


def test_negative_spread_upper_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,uniform,0.2,-0.1\n"
    )
    with pytest.raises(ValueError, match="spread_upper"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: negative min_abs
# ---------------------------------------------------------------------------

def test_negative_min_abs_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_proj_yd1,normal,0.3,0.3,-0.1\n"
    )
    with pytest.raises(ValueError, match="min_abs"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: min_abs on skew_normal / beta
# ---------------------------------------------------------------------------

def test_min_abs_on_skew_normal_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_proj_yd1,skew_normal,0.2,0.5,0.1\n"
    )
    with pytest.raises(ValueError, match="skew_normal"):
        load_distributions(csv, BASE_DICT)


def test_min_abs_on_beta_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper,min_abs\n"
        "crop_base_left1,beta,2.0,5.0,0.01\n"
    )
    with pytest.raises(ValueError, match="beta"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: asymmetric spread on symmetric distributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", ["normal", "truncated_normal", "lognormal"])
def test_asymmetric_spread_on_symmetric_distribution_raises(tmp_path, dist):
    csv = write_csv(tmp_path,
        f"parameter,distribution,spread_lower,spread_upper\n"
        f"crop_proj_yd1,{dist},0.2,0.5\n"
    )
    with pytest.raises(ValueError, match="symmetric"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Error: beta on non-fraction parameter
# ---------------------------------------------------------------------------

def test_beta_on_non_fraction_parameter_raises(tmp_path):
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,beta,2.0,5.0\n"  # crop_proj_yd1 has mean ~6, not in [0,1]
    )
    with pytest.raises(ValueError, match="beta"):
        load_distributions(csv, BASE_DICT)


# ---------------------------------------------------------------------------
# Multiple errors reported together
# ---------------------------------------------------------------------------

def test_multiple_errors_reported_together(tmp_path):
    """Two bad rows: both errors should appear in the same exception message."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "not_a_param,normal,0.3,0.3\n"
        "crop_proj_yd1,made_up_dist,0.3,0.3\n"
    )
    with pytest.raises(ValueError) as exc_info:
        load_distributions(csv, BASE_DICT)
    msg = str(exc_info.value)
    assert "not_a_param" in msg
    assert "made_up_dist" in msg


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_warning_base_zero_no_min_abs(tmp_path):
    """Base value of 0 without min_abs triggers a warning."""
    base = {**BASE_DICT, "zero_param": np.array([0.0, 0.0])}
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "zero_param,normal,0.3,0.3\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_distributions(csv, base)
    assert any("zero_param" in str(w.message) for w in caught)


def test_warning_fraction_parameter_can_breach_unit_interval(tmp_path):
    """Fraction parameter with a distribution that can breach [0,1] triggers a warning."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_base_left1,normal,0.3,0.3\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_distributions(csv, BASE_DICT)
    assert any("crop_base_left1" in str(w.message) for w in caught)


def test_warning_temp_without_min_abs(tmp_path):
    """temp parameter without min_abs triggers a warning."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "temp,normal,0.1,0.1\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_distributions(csv, BASE_DICT)
    assert any("temp" in str(w.message) for w in caught)


def test_warning_skew_normal_nearly_symmetric(tmp_path):
    """skew_normal with nearly equal spreads triggers a warning to use normal."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,skew_normal,0.300,0.301\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_distributions(csv, BASE_DICT)
    assert any("normal" in str(w.message).lower() for w in caught)


def test_warning_lognormal_high_sigma(tmp_path):
    """lognormal with spread > 0.5 triggers a warning about the approximation."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_proj_yd1,lognormal,0.8,0.8\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_distributions(csv, BASE_DICT)
    assert any("0.5" in str(w.message) for w in caught)


def test_no_error_raised_despite_warnings(tmp_path):
    """Rows that trigger warnings but no errors still return a valid spec."""
    csv = write_csv(tmp_path,
        "parameter,distribution,spread_lower,spread_upper\n"
        "crop_base_left1,normal,0.3,0.3\n"  # fraction + normal → warning only
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        specs = load_distributions(csv, BASE_DICT)
    assert "crop_base_left1" in specs
