"""Tests for monte_carlo/sampler.py (Issue 2).

Covers:
- Output list has length n_samples; each sample dict has same keys and array shapes as base
- Sampled values differ from base for perturbed keys (non-zero spread)
- Climate perturbation is multiplicative (ratio to base is a scalar uniform across months)
- Base dict is not mutated
- sample_soil_params: zero uncertainty (q05=q95=mean) → all samples equal mean
- sample_soil_params: non-zero uncertainty produces spread; clips Cy0 >= 0, clay in [0, 100]
- sample_climate_params: zero std → all samples equal mean
- sample_climate_params: non-zero std produces spread; rain clipped to >= 0
- Skew-normal with equal spread_lower/spread_upper produces draws centred on base
- All seven distribution types produce the correct number of samples
- Fraction parameters clamped to [0, 1] with a warning
"""

import warnings

import numpy as np
import pytest

from model.monte_carlo.distribution_handler import DistributionSpec
from model.monte_carlo.sampler import (
    draw_samples,
    sample_soil_params,
    sample_climate_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spec(dist, sl, su, min_abs=None):
    return DistributionSpec(
        parameter="dummy",
        distribution=dist,
        spread_lower=sl,
        spread_upper=su,
        min_abs=min_abs,
    )


class FakeSoilParams:
    def __init__(self, Cy0, clay, Cy0_q05, Cy0_q95, clay_q05, clay_q95):
        self.Cy0 = Cy0
        self.clay = clay
        self.Cy0_q05 = Cy0_q05
        self.Cy0_q95 = Cy0_q95
        self.clay_q05 = clay_q05
        self.clay_q95 = clay_q95


class FakeClimateData:
    def __init__(self, temperature, rain, evaporation,
                 temperature_std=None, rain_std=None, evaporation_std=None):
        self.temperature = np.array(temperature)
        self.rain = np.array(rain)
        self.evaporation = np.array(evaporation)
        self.temperature_std = np.zeros_like(self.temperature) if temperature_std is None else np.array(temperature_std)
        self.rain_std = np.zeros_like(self.rain) if rain_std is None else np.array(rain_std)
        self.evaporation_std = np.zeros_like(self.evaporation) if evaporation_std is None else np.array(evaporation_std)


BASE_DICT = {
    "crop_proj_yd1": np.array([5.0, 6.0, 7.0]),    # vector, non-fraction
    "base_lit_qty1": np.array([2.0, 2.0, 2.0]),    # vector, non-fraction
    "Rain":          np.array([80.0] * 12),          # climate key, vector
    "Temp":          np.array([20.0] * 12),          # climate key, vector
    "evap":          np.array([50.0] * 12),          # climate key, vector
    "crop_base_left1": np.array([0.3, 0.3, 0.3]),  # fraction parameter
}


# ---------------------------------------------------------------------------
# draw_samples — output structure
# ---------------------------------------------------------------------------

def test_output_length():
    specs = {"crop_proj_yd1": make_spec("normal", 0.1, 0.1)}
    samples = draw_samples(BASE_DICT, specs, n_samples=10, seed=0)
    assert len(samples) == 10


def test_output_keys_match_base():
    specs = {"crop_proj_yd1": make_spec("normal", 0.1, 0.1)}
    samples = draw_samples(BASE_DICT, specs, n_samples=5, seed=0)
    for sample in samples:
        assert set(sample.keys()) == set(BASE_DICT.keys())


def test_output_shapes_match_base():
    specs = {"crop_proj_yd1": make_spec("normal", 0.1, 0.1)}
    samples = draw_samples(BASE_DICT, specs, n_samples=5, seed=0)
    for sample in samples:
        for key in BASE_DICT:
            assert np.asarray(sample[key]).shape == np.asarray(BASE_DICT[key]).shape


# ---------------------------------------------------------------------------
# draw_samples — values differ from base
# ---------------------------------------------------------------------------

def test_perturbed_values_differ_from_base():
    """With large spread, drawn values should differ from base (across N=100 samples)."""
    specs = {"crop_proj_yd1": make_spec("normal", 0.5, 0.5)}
    samples = draw_samples(BASE_DICT, specs, n_samples=100, seed=42)
    perturbed = [s["crop_proj_yd1"] for s in samples]
    base = BASE_DICT["crop_proj_yd1"]
    assert not all(np.allclose(p, base) for p in perturbed)


def test_vector_draws_are_element_wise():
    """Each element of a yearly vector is drawn independently using its own value as base.

    crop_proj_yd1 = [5.0, 6.0, 7.0] with CV=0.2 gives stds [1.0, 1.2, 1.4].
    Across many samples, the variance of each position should reflect its own base value,
    not a shared scalar broadcast from the vector mean.
    """
    specs = {"crop_proj_yd1": make_spec("normal", 0.2, 0.2)}
    samples = draw_samples(BASE_DICT, specs, n_samples=1000, seed=0)
    # Extract each position across all samples
    pos0 = np.array([s["crop_proj_yd1"][0] for s in samples])  # base=5.0, expected std≈1.0
    pos2 = np.array([s["crop_proj_yd1"][2] for s in samples])  # base=7.0, expected std≈1.4
    # std at position 2 should be meaningfully larger than at position 0
    assert np.std(pos2) > np.std(pos0) * 1.1, (
        f"Expected element-wise draws: std[pos2]={np.std(pos2):.3f} should exceed "
        f"std[pos0]={np.std(pos0):.3f} (bases are 7.0 vs 5.0)"
    )


def test_unperturbed_keys_unchanged():
    """Keys not in distributions are copied unchanged from the base."""
    specs = {"crop_proj_yd1": make_spec("normal", 0.1, 0.1)}
    samples = draw_samples(BASE_DICT, specs, n_samples=5, seed=0)
    for sample in samples:
        np.testing.assert_array_equal(sample["base_lit_qty1"], BASE_DICT["base_lit_qty1"])


# ---------------------------------------------------------------------------
# draw_samples — base dict not mutated
# ---------------------------------------------------------------------------

def test_base_dict_not_mutated():
    original = {k: np.copy(v) for k, v in BASE_DICT.items()}
    specs = {"crop_proj_yd1": make_spec("normal", 0.3, 0.3)}
    draw_samples(BASE_DICT, specs, n_samples=20, seed=0)
    for key in original:
        np.testing.assert_array_equal(BASE_DICT[key], original[key])


# ---------------------------------------------------------------------------
# draw_samples — climate perturbation is multiplicative
# ---------------------------------------------------------------------------

def test_climate_perturbation_is_multiplicative():
    """Rain perturbation: the ratio of drawn to base is a scalar uniform across months."""
    specs = {"Rain": make_spec("normal", 0.2, 0.2)}
    samples = draw_samples(BASE_DICT, specs, n_samples=30, seed=7)
    base_rain = BASE_DICT["Rain"]
    for sample in samples:
        rain = sample["Rain"]
        ratios = rain / base_rain
        # All ratios should be equal (within floating point) — the same scalar was applied
        assert np.allclose(ratios, ratios[0]), f"Non-uniform ratio across months: {ratios}"


def test_climate_evap_perturbation_is_multiplicative():
    specs = {"evap": make_spec("uniform", 0.1, 0.2)}
    samples = draw_samples(BASE_DICT, specs, n_samples=20, seed=3)
    base_evap = BASE_DICT["evap"]
    for sample in samples:
        evap = sample["evap"]
        ratios = evap / base_evap
        assert np.allclose(ratios, ratios[0])


# ---------------------------------------------------------------------------
# draw_samples — all seven distribution types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist,sl,su", [
    ("normal",           0.2, 0.2),
    ("truncated_normal", 0.2, 0.2),
    ("lognormal",        0.3, 0.3),
    ("uniform",          0.1, 0.2),
    ("triangular",       0.1, 0.2),
    ("skew_normal",      0.2, 0.5),
    ("beta",             2.0, 5.0),
])
def test_all_distributions_produce_n_samples(dist, sl, su):
    if dist == "beta":
        param = "crop_base_left1"  # fraction parameter required for beta
    else:
        param = "crop_proj_yd1"
    specs = {param: make_spec(dist, sl, su)}
    samples = draw_samples(BASE_DICT, specs, n_samples=10, seed=1)
    assert len(samples) == 10
    for sample in samples:
        assert param in sample
        arr = np.asarray(sample[param])
        assert arr.shape == np.asarray(BASE_DICT[param]).shape


# ---------------------------------------------------------------------------
# draw_samples — skew-normal with equal spreads
# ---------------------------------------------------------------------------

def test_skew_normal_equal_spreads_centred_on_base():
    """skew_normal with spread_lower == spread_upper should produce draws centred near base."""
    specs = {"crop_proj_yd1": make_spec("skew_normal", 0.3, 0.3)}
    samples = draw_samples(BASE_DICT, specs, n_samples=500, seed=42)
    base_mean = float(np.mean(BASE_DICT["crop_proj_yd1"]))
    drawn_values = [float(s["crop_proj_yd1"][0]) for s in samples]
    sample_mean = np.mean(drawn_values)
    # Mean of draws should be close to base (within 20% relative tolerance for N=500)
    assert abs(sample_mean - base_mean) / base_mean < 0.20, (
        f"skew_normal(equal spreads) mean {sample_mean:.2f} far from base {base_mean:.2f}"
    )


# ---------------------------------------------------------------------------
# draw_samples — min_abs floor
# ---------------------------------------------------------------------------

def test_min_abs_floor_normal():
    """With a tiny base value, min_abs dominates the std and produces meaningful spread."""
    tiny_base = {**BASE_DICT, "base_lit_qty1": np.array([0.001, 0.001, 0.001])}
    specs = {"base_lit_qty1": make_spec("normal", 0.1, 0.1, min_abs=0.5)}
    samples = draw_samples(tiny_base, specs, n_samples=100, seed=0)
    values = [float(s["base_lit_qty1"][0]) for s in samples]
    # std should be dominated by min_abs=0.5, not by base * 0.1 = 0.0001
    assert np.std(values) > 0.1, "min_abs floor should produce meaningful spread"


# ---------------------------------------------------------------------------
# draw_samples — fraction parameter clamping
# ---------------------------------------------------------------------------

def test_fraction_parameter_clamped_with_warning():
    """Normal distribution on a fraction parameter with large spread triggers clamping and a warning."""
    # crop_base_left1 is in [0.3, 0.3, 0.3]; normal with CV=2.0 will easily breach [0, 1]
    specs = {"crop_base_left1": make_spec("normal", 2.0, 2.0)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = draw_samples(BASE_DICT, specs, n_samples=100, seed=0)
    # All sampled values should be in [0, 1]
    for sample in samples:
        arr = sample["crop_base_left1"]
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0), f"Fraction breach not clamped: {arr}"
    # At least one warning should mention the parameter
    assert any("crop_base_left1" in str(w.message) for w in caught), (
        "Expected a clamping warning for fraction parameter breach"
    )


# ---------------------------------------------------------------------------
# sample_soil_params — zero uncertainty
# ---------------------------------------------------------------------------

def test_sample_soil_params_zero_uncertainty():
    """When q05 == q95 == mean, all samples equal the mean."""
    soil = FakeSoilParams(Cy0=5.0, clay=30.0, Cy0_q05=5.0, Cy0_q95=5.0,
                          clay_q05=30.0, clay_q95=30.0)
    rng = np.random.default_rng(0)
    samples = sample_soil_params(soil, n_samples=20, rng=rng)
    assert len(samples) == 20
    for s in samples:
        assert s["Cy0"] == pytest.approx(5.0)
        assert s["clay"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# sample_soil_params — non-zero uncertainty
# ---------------------------------------------------------------------------

def test_sample_soil_params_nonzero_spread():
    """Non-zero quantiles produce spread in draws."""
    soil = FakeSoilParams(Cy0=5.0, clay=30.0, Cy0_q05=3.0, Cy0_q95=7.0,
                          clay_q05=25.0, clay_q95=35.0)
    rng = np.random.default_rng(1)
    samples = sample_soil_params(soil, n_samples=100, rng=rng)
    cy0_vals = [s["Cy0"] for s in samples]
    clay_vals = [s["clay"] for s in samples]
    assert np.std(cy0_vals) > 0.1
    assert np.std(clay_vals) > 0.1


def test_sample_soil_params_cy0_clipped_to_nonnegative():
    """Cy0 draws are clipped to >= 0 even when quantiles are very close to 0."""
    # Cy0=0.1 with wide quantiles: many draws would be negative without clipping
    soil = FakeSoilParams(Cy0=0.1, clay=30.0, Cy0_q05=0.0, Cy0_q95=5.0,
                          clay_q05=25.0, clay_q95=35.0)
    rng = np.random.default_rng(2)
    samples = sample_soil_params(soil, n_samples=200, rng=rng)
    assert all(s["Cy0"] >= 0.0 for s in samples)


def test_sample_soil_params_clay_clipped_to_valid_range():
    """Clay draws are clipped to [0, 100]."""
    soil = FakeSoilParams(Cy0=5.0, clay=98.0, Cy0_q05=4.0, Cy0_q95=6.0,
                          clay_q05=90.0, clay_q95=106.0)  # q95 > 100 is invalid in real data, but tests clip
    rng = np.random.default_rng(3)
    samples = sample_soil_params(soil, n_samples=200, rng=rng)
    assert all(0.0 <= s["clay"] <= 100.0 for s in samples)


# ---------------------------------------------------------------------------
# sample_climate_params — zero std
# ---------------------------------------------------------------------------

def test_sample_climate_params_zero_std():
    """When all stds are 0, all samples equal the means."""
    climate = FakeClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[50.0] * 12,
    )
    rng = np.random.default_rng(0)
    samples = sample_climate_params(climate, n_samples=10, rng=rng)
    assert len(samples) == 10
    for s in samples:
        np.testing.assert_array_equal(s["Temp"], climate.temperature)
        np.testing.assert_array_equal(s["Rain"], climate.rain)
        np.testing.assert_array_equal(s["evap"], climate.evaporation)


# ---------------------------------------------------------------------------
# sample_climate_params — non-zero std
# ---------------------------------------------------------------------------

def test_sample_climate_params_nonzero_std():
    """Non-zero std produces spread in temperature draws."""
    climate = FakeClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[50.0] * 12,
        temperature_std=[2.0] * 12,
        rain_std=[10.0] * 12,
        evaporation_std=[5.0] * 12,
    )
    rng = np.random.default_rng(4)
    samples = sample_climate_params(climate, n_samples=100, rng=rng)
    temp_vals = np.array([s["Temp"][0] for s in samples])
    assert np.std(temp_vals) > 0.1


def test_sample_climate_params_rain_clipped():
    """Rain draws are clipped to >= 0."""
    climate = FakeClimateData(
        temperature=[20.0] * 12,
        rain=[1.0] * 12,   # very small mean — many draws would be negative
        evaporation=[50.0] * 12,
        rain_std=[10.0] * 12,  # very wide std
    )
    rng = np.random.default_rng(5)
    samples = sample_climate_params(climate, n_samples=200, rng=rng)
    assert all(np.all(s["Rain"] >= 0.0) for s in samples)


def test_sample_climate_params_evap_clipped():
    """Evaporation draws are clipped to >= 0."""
    climate = FakeClimateData(
        temperature=[20.0] * 12,
        rain=[80.0] * 12,
        evaporation=[1.0] * 12,
        evaporation_std=[10.0] * 12,
    )
    rng = np.random.default_rng(6)
    samples = sample_climate_params(climate, n_samples=200, rng=rng)
    assert all(np.all(s["evap"] >= 0.0) for s in samples)
