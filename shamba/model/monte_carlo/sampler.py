"""Draw N perturbed input dicts from a base dict and validated distribution specs.

Also provides sample_soil_params() and sample_climate_params() for drawing from
uncertainty stored in SoilParamsData and ClimateData (Issues P1/P2).
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import scipy.stats
import scipy.optimize

from model.monte_carlo.distribution_handler import DistributionSpec

# Climate parameter keys — perturbation is applied as a multiplicative scalar
# to preserve the seasonal structure of the monthly vector.
_CLIMATE_KEYS = frozenset({"Temp", "Rain", "evap"})

# Fraction parameters: if any drawn value breaches [0, 1], clamp and warn.
# The check is based on the base value, as determined by distribution_handler.
# We re-derive this at draw time via the base value itself.


def _base_mean(base_value) -> float:
    """Scalar mean of a base value — handles scalars and numpy arrays."""
    arr = np.asarray(base_value, dtype=float).ravel()
    return float(np.mean(arr))


def _is_fraction_parameter(base_value) -> bool:
    """True if every element of the base value lies within [0, 1]."""
    arr = np.asarray(base_value, dtype=float).ravel()
    return bool(np.all((arr >= 0.0) & (arr <= 1.0)))


def _draw_one(
    spec: DistributionSpec,
    base_mean: float,
    rng: np.random.Generator,
) -> float:
    """Draw a single scalar perturbation from the distribution described by spec.

    The return value is an absolute draw in the same units as the base value,
    scaled to the supplied base_mean. The caller is responsible for how the
    draw is applied: climate keys use it to derive a multiplicative ratio;
    non-climate vector parameters call this once per element using each
    element's own value as base_mean, so that the spread scales with the
    local value rather than the vector mean.

    Edge cases
    ----------
    base_mean == 0:
        Relative spread (CV-based) is undefined when the mean is zero. For
        distributions that scale spread as ``base_mean * spread_lower``, the
        effective std collapses to zero regardless of the spread parameter.
        If ``min_abs`` is supplied it still applies, giving a non-zero floor.
        If not, the draw returns ``base_mean`` (i.e. 0) deterministically.
        ``distribution_handler`` warns when a zero base value is paired with
        no ``min_abs``, so this path should rarely be reached silently.

    effective_std == 0 (lognormal):
        ``scipy.stats.lognorm`` with ``scale=0`` is degenerate and raises an
        error. We short-circuit and return ``base_mean`` instead. This also
        avoids a division-by-zero when computing ``effective_sigma = std / mean``.

    truncated_normal near zero:
        The lower bound is expressed in standardised units as
        ``a = -base_mean / effective_std``. When ``base_mean`` is large relative
        to ``effective_std``, ``a`` is a large negative number and truncation has
        negligible effect. When ``base_mean`` is small (close to zero), ``a``
        approaches 0, the left tail is heavily truncated, and the resulting
        distribution is visibly right-skewed even though the spec is symmetric.
        This asymmetry is a physical consequence of the non-negativity constraint,
        not a bug.

    Args:
        spec: validated DistributionSpec from distribution_handler.
        base_mean: mean of the base vector (or the scalar base value).
        rng: numpy random Generator for reproducible draws.

    Returns:
        float: a single drawn value.
    """
    dist = spec.distribution
    sl = spec.spread_lower
    su = spec.spread_upper
    min_abs = spec.min_abs

    if dist == "normal":
        effective_std = max(base_mean * sl, min_abs) if min_abs is not None else base_mean * sl
        return float(rng.normal(loc=base_mean, scale=effective_std))

    if dist == "truncated_normal":
        effective_std = max(base_mean * sl, min_abs) if min_abs is not None else base_mean * sl
        if effective_std == 0.0:
            return base_mean
        a = -base_mean / effective_std  # lower clip at 0
        return float(scipy.stats.truncnorm.rvs(a=a, b=np.inf, loc=base_mean, scale=effective_std, random_state=rng))

    if dist == "lognormal":
        effective_std = max(base_mean * sl, min_abs) if min_abs is not None else base_mean * sl
        if base_mean == 0.0 or effective_std == 0.0:
            return base_mean
        # sigma ≈ std / mean (normal approximation, accurate for sigma < ~0.5)
        effective_sigma = effective_std / base_mean
        return float(scipy.stats.lognorm.rvs(s=effective_sigma, scale=base_mean, random_state=rng))

    if dist == "uniform":
        effective_lower_hw = max(base_mean * sl, min_abs) if min_abs is not None else base_mean * sl
        effective_upper_hw = max(base_mean * su, min_abs) if min_abs is not None else base_mean * su
        low = base_mean - effective_lower_hw
        high = base_mean + effective_upper_hw
        return float(rng.uniform(low=low, high=high))

    if dist == "triangular":
        effective_lower_hw = max(base_mean * sl, min_abs) if min_abs is not None else base_mean * sl
        effective_upper_hw = max(base_mean * su, min_abs) if min_abs is not None else base_mean * su
        low = base_mean - effective_lower_hw
        high = base_mean + effective_upper_hw
        # scipy.stats.triang: c is the fractional position of the mode within [low, high]
        width = high - low
        if width == 0.0:
            return base_mean
        c = (base_mean - low) / width
        return float(scipy.stats.triang.rvs(c=c, loc=low, scale=width, random_state=rng))

    if dist == "skew_normal":
        return _draw_skew_normal(sl, su, base_mean, rng)

    if dist == "beta":
        # spread_lower = α, spread_upper = β — parameters taken directly from literature.
        # Domain is naturally [0, 1]; base value is not used to rescale.
        return float(scipy.stats.beta.rvs(a=sl, b=su, random_state=rng))

    raise ValueError(f"Unsupported distribution: '{dist}'")  # should never reach here


def _draw_skew_normal(
    spread_lower: float,
    spread_upper: float,
    base_mean: float,
    rng: np.random.Generator,
) -> float:
    """Draw from a skew-normal distribution back-calculated to match two CV half-widths.

    The skewnorm distribution has three parameters (a, loc, scale):
      - E[X]   = loc + scale * delta * sqrt(2/pi)      where delta = a / sqrt(1 + a^2)
      - Var[X] = scale^2 * (1 - 2*delta^2/pi)

    We target:
      - Mean    ≈ base_mean
      - Left σ  ≈ base_mean * spread_lower
      - Right σ ≈ base_mean * spread_upper

    Strategy:
      total_sigma = (spread_lower + spread_upper) / 2 * base_mean
      asymmetry_ratio = (spread_upper - spread_lower) / (spread_upper + spread_lower)
        (positive → right-skewed, negative → left-skewed)
      Solve numerically for `a` such that the half-width ratio matches.
    """
    total_sigma = (spread_lower + spread_upper) / 2.0 * base_mean
    if total_sigma == 0.0:
        return base_mean

    asymmetry_ratio = (spread_upper - spread_lower) / (spread_upper + spread_lower)

    def _skewnorm_half_width_ratio(a):
        """Target: (right_hw - left_hw) / (right_hw + left_hw) = asymmetry_ratio.

        For skewnorm, the distribution is characterised by the skewness parameter a.
        We approximate the ratio by the standardised skewnorm's asymmetry in its
        half-widths relative to the mean — found numerically.
        """
        delta = a / np.sqrt(1.0 + a ** 2)
        mean_shift = delta * np.sqrt(2.0 / np.pi)
        var = 1.0 - 2.0 * delta ** 2 / np.pi
        if var <= 0:
            return -asymmetry_ratio  # degenerate

        # For a standardised skewnorm(a, loc=0, scale=1), approximate left/right
        # half-widths as distance from mean to 1-sigma quantiles.
        std_sn = np.sqrt(var)
        mean_sn = mean_shift
        # right half-width ≈ +1 std from mean; left ≈ 1 std below mean
        # This is a symmetric-std approximation — good enough for back-calculation
        # since we only use `a` to steer skewness direction and magnitude.
        right_hw = std_sn
        left_hw = std_sn
        # Correction: skewnorm with a > 0 has a longer right tail.
        # A more accurate proxy: use the difference in (median - mean) vs (mean - median).
        # Numerically: compute the actual 1-sigma quantiles of skewnorm(a, 0, 1).
        q_low = scipy.stats.skewnorm.ppf(0.1587, a)   # ≈ mean - 1σ for normal
        q_high = scipy.stats.skewnorm.ppf(0.8413, a)  # ≈ mean + 1σ for normal
        actual_right = q_high - mean_sn
        actual_left = mean_sn - q_low
        denom = actual_right + actual_left
        if denom == 0:
            return 0.0
        return (actual_right - actual_left) / denom

    # Find a in [-10, 10] such that the half-width ratio matches the target.
    try:
        a_opt = scipy.optimize.brentq(
            lambda a: _skewnorm_half_width_ratio(a) - asymmetry_ratio,
            -10.0,
            10.0,
            xtol=1e-6,
        )
    except ValueError:
        # brentq could not bracket — fall back to a=0 (symmetric normal behaviour)
        warnings.warn(
            "skew_normal back-calculation did not converge; falling back to symmetric normal.",
            UserWarning,
            stacklevel=4,
        )
        a_opt = 0.0

    # Compute loc and scale so that E[X] = base_mean and scale ≈ total_sigma.
    delta = a_opt / np.sqrt(1.0 + a_opt ** 2)
    mean_shift = delta * np.sqrt(2.0 / np.pi)
    var_factor = 1.0 - 2.0 * delta ** 2 / np.pi
    if var_factor <= 0.0:
        var_factor = 1e-9
    scale = total_sigma / np.sqrt(var_factor)
    loc = base_mean - scale * mean_shift

    return float(scipy.stats.skewnorm.rvs(a=a_opt, loc=loc, scale=scale, random_state=rng))


def draw_samples(
    base_input_dict: Dict,
    distributions: Dict[str, DistributionSpec],
    n_samples: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Draw N perturbed copies of base_input_dict.

    For vector parameters, one draw is made independently per element using
    that element's own value as the local base. This means the spread scales
    with the local value: a 20% CV on crop yields of [5.0, 6.0, 7.0] gives
    stds of [1.0, 1.2, 1.4] respectively — higher-yield years have wider
    absolute uncertainty but the same relative uncertainty.

    Climate keys (Temp, Rain, evap) are the exception: a single multiplicative
    scalar is drawn from the vector mean and applied to the whole monthly vector,
    preserving the seasonal structure. A climate uncertainty of 10% shifts all
    months up or down together rather than perturbing each month independently.

    The base dict is never mutated.

    Args:
        base_input_dict: validated input dict for the run.
        distributions: dict of parameter name → DistributionSpec (from load_distributions).
        n_samples: number of samples to draw.
        seed: optional integer seed for reproducibility. Ignored if rng is provided.
        rng: optional numpy Generator. If supplied, it is used directly (and
            seed is ignored), allowing the caller to thread a single RNG across
            multiple sampling steps.

    Returns:
        list of n_samples dicts, each a perturbed copy of base_input_dict.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        sample = dict(base_input_dict)  # shallow copy — scalars and arrays unchanged

        for param, spec in distributions.items():
            base_value = base_input_dict[param]
            arr = np.asarray(base_value, dtype=float)
            is_fraction = _is_fraction_parameter(base_value)

            if param in _CLIMATE_KEYS:
                # Exception: draw one multiplicative scalar from the vector mean and
                # apply to the whole monthly vector, preserving the seasonal structure.
                bm = _base_mean(base_value)
                drawn = _draw_one(spec, bm, rng)
                ratio = drawn / bm if bm != 0.0 else 1.0
                sample[param] = arr * ratio
            else:
                # Draw independently per element, scaling spread to each element's
                # own value. This preserves the relative uncertainty across years:
                # a 20% CV on [5, 6, 7] gives stds of [1.0, 1.2, 1.4].
                perturbed = np.array([_draw_one(spec, float(elem), rng) for elem in arr.ravel()])
                perturbed = perturbed.reshape(arr.shape)

                # Clamp fraction parameters to [0, 1].
                if is_fraction:
                    clipped = np.clip(perturbed, 0.0, 1.0)
                    if not np.array_equal(clipped, perturbed):
                        warnings.warn(
                            f"Parameter '{param}': one or more drawn values breach [0, 1] "
                            f"and have been clamped. Consider using 'beta' or 'truncated_normal'.",
                            UserWarning,
                            stacklevel=2,
                        )
                    perturbed = clipped

                sample[param] = perturbed

        samples.append(sample)

    return samples


def sample_soil_params(
    soil,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Draw N soil parameter dicts from the uncertainty stored in SoilParamsData.

    Fits a normal distribution to the stored quantiles:
      sigma = (q95 - q05) / (2 * 1.645)
    If q05 == q95 == mean, sigma = 0 and all samples equal the mean.

    Cy0 is clipped to >= 0; clay is clipped to [0, 100].

    Args:
        soil: SoilParamsData object with Cy0, clay and their quantile fields.
        n_samples: number of samples to draw.
        rng: numpy random Generator.

    Returns:
        list of n_samples dicts with keys: 'Cy0', 'clay'.
    """
    # Infer sigma from Q0.05/Q0.95 (1.645 is the z-score for the 95th percentile).
    cy0_sigma = (soil.Cy0_q95 - soil.Cy0_q05) / (2.0 * 1.645)
    clay_sigma = (soil.clay_q95 - soil.clay_q05) / (2.0 * 1.645)

    if cy0_sigma == 0.0:
        cy0_draws = np.full(n_samples, soil.Cy0)
    else:
        cy0_draws = rng.normal(loc=soil.Cy0, scale=cy0_sigma, size=n_samples)
        cy0_draws = np.clip(cy0_draws, 0.0, None)

    if clay_sigma == 0.0:
        clay_draws = np.full(n_samples, soil.clay)
    else:
        clay_draws = rng.normal(loc=soil.clay, scale=clay_sigma, size=n_samples)
        clay_draws = np.clip(clay_draws, 0.0, 100.0)

    return [
        {"Cy0": float(cy0_draws[i]), "clay": float(clay_draws[i])}
        for i in range(n_samples)
    ]


def sample_climate_params(
    climate,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Draw N climate parameter dicts from the uncertainty stored in ClimateData.

    Draws from Normal(mean, std) for each monthly value independently.
    Rain and evaporation are clipped to >= 0. If all stds are 0, all samples
    equal the means.

    Args:
        climate: ClimateData object with temperature, rain, evaporation and their _std arrays.
        n_samples: number of samples to draw.
        rng: numpy random Generator.

    Returns:
        list of n_samples dicts with keys: 'Temp', 'Rain', 'evap'.
        Each value is a numpy array of the same length as the base climate vectors.
    """
    temp_mean = climate.temperature
    temp_std = climate.temperature_std
    rain_mean = climate.rain
    rain_std = climate.rain_std
    evap_mean = climate.evaporation
    evap_std = climate.evaporation_std

    results = []
    for _ in range(n_samples):
        if np.all(temp_std == 0.0):
            temp_draw = temp_mean.copy()
        else:
            temp_draw = rng.normal(loc=temp_mean, scale=temp_std)

        if np.all(rain_std == 0.0):
            rain_draw = rain_mean.copy()
        else:
            rain_draw = np.clip(rng.normal(loc=rain_mean, scale=rain_std), 0.0, None)

        if np.all(evap_std == 0.0):
            evap_draw = evap_mean.copy()
        else:
            evap_draw = np.clip(rng.normal(loc=evap_mean, scale=evap_std), 0.0, None)

        results.append({"Temp": temp_draw, "Rain": rain_draw, "evap": evap_draw})

    return results
