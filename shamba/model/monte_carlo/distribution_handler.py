"""Read and validate a user-supplied distributions CSV file for Monte Carlo sampling."""

import warnings
from typing import Dict, List, Optional, NamedTuple

import numpy as np
import pandas as pd


SUPPORTED_DISTRIBUTIONS = frozenset({
    "normal",
    "truncated_normal",
    "lognormal",
    "uniform",
    "triangular",
    "skew_normal",
    "beta",
})

# These distributions take a single spread value: spread_lower must equal spread_upper.
_SYMMETRIC_DISTRIBUTIONS = frozenset({
    "normal",            # spread is CV (σ/μ); symmetric around base
    "truncated_normal",  # same CV as normal; asymmetry is fixed (truncated at 0, not user-controlled)
    "lognormal",         # spread is the σ shape parameter; median = base, but right-skewed by construction
})

# min_abs is not meaningful for these distributions and is rejected.
_NO_MIN_ABS_DISTRIBUTIONS = frozenset({
    "skew_normal",  # asymmetry is controlled by spread_lower/spread_upper; clip draws at 0 instead
    "beta",         # spread_lower/spread_upper are α/β shape params that fully define the domain
})

# These distributions can produce values outside [0, 1] for fraction parameters.
_CAN_BREACH_UNIT_INTERVAL = frozenset({
    "normal",      # unbounded; can go negative or above 1
    "lognormal",   # bounded below at 0, but unbounded above; can exceed 1
    "uniform",     # bounds are base ± spread; can exceed [0,1] with large spread
    "triangular",  # same as uniform — bounds are user-controlled and may breach [0,1]
})


class DistributionSpec(NamedTuple):
    """Validated specification for a single parameter distribution.

    The sampler uses this to construct the actual scipy frozen distribution,
    applying the min_abs floor against the base value at draw time.
    """
    parameter: str
    distribution: str
    spread_lower: float
    spread_upper: float
    min_abs: Optional[float]  # None if not specified in the CSV


def _base_mean(base_value) -> float:
    """Scalar mean of a base value — handles both scalars and numpy arrays."""
    arr = np.asarray(base_value, dtype=float).ravel()
    return float(np.mean(arr))


def _is_fraction_parameter(base_value) -> bool:
    """True if every element of the base value lies within [0, 1]."""
    arr = np.asarray(base_value, dtype=float).ravel()
    return bool(np.all((arr >= 0.0) & (arr <= 1.0)))


def _parse_float(value, label: str, errors: list) -> Optional[float]:
    """Parse a float from a CSV cell; append to errors and return None on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        errors.append(f"{label} must be a number.")
        return None


def load_distributions(
    path: str,
    base_input_dict: Dict,
) -> Dict[str, DistributionSpec]:
    """Read and validate a distributions CSV file.

    CSV columns (header row required):
        parameter      — name of the input dict key to perturb
        distribution   — one of: normal, truncated_normal, lognormal, uniform,
                         triangular, skew_normal, beta
        spread_lower   — lower spread parameter (distribution-specific meaning)
        spread_upper   — upper spread parameter (must equal spread_lower for
                         symmetric distributions)
        min_abs        — optional absolute floor on the effective spread;
                         not supported for skew_normal or beta

    For symmetric distributions (normal, truncated_normal, lognormal),
    spread_lower and spread_upper must be equal. Use skew_normal for asymmetric
    distributions.

    All errors are collected and raised together as a single ValueError.
    Warnings are emitted via warnings.warn() for edge cases that are valid but
    potentially unintentional.

    Args:
        path: path to the distributions CSV file.
        base_input_dict: the validated merged input dict for the run. Used to
            check parameter names exist and to inspect base values for warnings.

    Returns:
        dict mapping parameter name → DistributionSpec for rows that passed
        validation. Only returned if there are no errors across all rows.

    Raises:
        ValueError: if the file cannot be read, required columns are missing,
            or any row fails validation. All errors reported together.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read distributions file '{path}': {e}")

    required_cols = {"parameter", "distribution", "spread_lower", "spread_upper"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Distributions file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    all_errors: List[str] = []
    specs: Dict[str, DistributionSpec] = {}

    for idx, row in df.iterrows():
        parameter = str(row["parameter"]).strip()
        row_label = f"Row {idx + 1} (parameter '{parameter}')"
        row_errors: List[str] = []

        # --- Validate parameter name first — skip remaining checks if unknown ---
        if parameter not in base_input_dict:
            all_errors.append(
                f"{row_label}: '{parameter}' is not a recognised input parameter."
            )
            continue

        distribution = str(row["distribution"]).strip().lower()

        spread_lower = _parse_float(row["spread_lower"], f"{row_label}: spread_lower", row_errors)
        # If spread_upper is absent or empty, copy from spread_lower (convenience for symmetric distributions).
        raw_upper = row["spread_upper"]
        if pd.isna(raw_upper) or str(raw_upper).strip() == "":
            spread_upper = spread_lower
        else:
            spread_upper = _parse_float(raw_upper, f"{row_label}: spread_upper", row_errors)

        min_abs: Optional[float] = None
        if "min_abs" in df.columns and pd.notna(row.get("min_abs")):
            min_abs = _parse_float(row["min_abs"], f"{row_label}: min_abs", row_errors)

        # If we couldn't parse the numeric fields, skip remaining checks for this row.
        if spread_lower is None or spread_upper is None:
            all_errors.extend(row_errors)
            continue

        # --- Validate distribution type ---
        if distribution not in SUPPORTED_DISTRIBUTIONS:
            row_errors.append(
                f"{row_label}: '{distribution}' is not a supported distribution. "
                f"Choose from: {', '.join(sorted(SUPPORTED_DISTRIBUTIONS))}."
            )
            # Can't do distribution-specific checks without a valid type.
            all_errors.extend(row_errors)
            continue

        # --- Validate spread values ---
        if spread_lower <= 0:
            row_errors.append(f"{row_label}: spread_lower must be greater than 0.")
        if spread_upper <= 0:
            row_errors.append(f"{row_label}: spread_upper must be greater than 0.")

        # --- Validate min_abs ---
        if min_abs is not None and min_abs < 0:
            row_errors.append(f"{row_label}: min_abs must be non-negative.")

        if min_abs is not None and distribution in _NO_MIN_ABS_DISTRIBUTIONS:
            row_errors.append(
                f"{row_label}: min_abs is not supported for '{distribution}'. "
                + (
                    "For skew_normal, draws below 0 are clipped in the sampler."
                    if distribution == "skew_normal"
                    else "For beta, spread_lower and spread_upper are the α and β shape parameters."
                )
            )

        # --- Symmetric distribution check ---
        if distribution in _SYMMETRIC_DISTRIBUTIONS and abs(spread_lower - spread_upper) > 1e-9:
            row_errors.append(
                f"{row_label}: '{distribution}' is symmetric — spread_lower and "
                f"spread_upper must be equal. Use 'skew_normal' for an asymmetric distribution."
            )

        # --- beta: only valid for fraction parameters ---
        if distribution == "beta" and parameter in base_input_dict:
            base_value = base_input_dict[parameter]
            if not _is_fraction_parameter(base_value):
                base_m = _base_mean(base_value)
                row_errors.append(
                    f"{row_label}: 'beta' is only valid for parameters whose base value "
                    f"lies in [0, 1]. '{parameter}' has base mean {base_m:.4g}."
                )

        # --- Warnings (only if no errors so far for this row) ---
        if not row_errors and parameter in base_input_dict:
            base_value = base_input_dict[parameter]
            base_m = _base_mean(base_value)

            if base_m == 0.0 and min_abs is None:
                warnings.warn(
                    f"{row_label}: base value for '{parameter}' is 0 — relative spread "
                    f"is undefined without a min_abs floor.",
                    UserWarning,
                    stacklevel=2,
                )

            if (
                min_abs is not None
                and base_m != 0.0
                and min_abs > base_m * spread_lower
            ):
                warnings.warn(
                    f"{row_label}: min_abs ({min_abs}) exceeds base × spread_lower "
                    f"({base_m * spread_lower:.4g}) — the floor will always dominate.",
                    UserWarning,
                    stacklevel=2,
                )

            if _is_fraction_parameter(base_value) and distribution in _CAN_BREACH_UNIT_INTERVAL:
                warnings.warn(
                    f"{row_label}: '{distribution}' can produce values outside [0, 1] "
                    f"for fraction parameter '{parameter}'. "
                    f"Consider 'beta' or 'truncated_normal' instead.",
                    UserWarning,
                    stacklevel=2,
                )

            if distribution == "skew_normal":
                total = spread_lower + spread_upper
                if total > 0 and abs(spread_upper - spread_lower) / total < 0.05:
                    warnings.warn(
                        f"{row_label}: spread_lower and spread_upper are nearly equal "
                        f"for 'skew_normal' — 'normal' would be simpler.",
                        UserWarning,
                        stacklevel=2,
                    )

            if distribution == "lognormal":
                # spread_lower is the σ shape parameter for lognormal
                effective_sigma = spread_lower
                if min_abs is not None and base_m > 0:
                    effective_sigma = max(base_m * spread_lower, min_abs) / base_m
                if effective_sigma > 0.5:
                    warnings.warn(
                        f"{row_label}: lognormal effective sigma ({effective_sigma:.3f}) "
                        f"is > 0.5 — the normal approximation breaks down at this spread.",
                        UserWarning,
                        stacklevel=2,
                    )

            if parameter == "temp" and min_abs is None:
                warnings.warn(
                    f"{row_label}: 'temp' uses a relative spread, but temperature can "
                    f"be near 0°C where relative spread is physically odd. "
                    f"Consider adding a min_abs floor (e.g. 1.0°C).",
                    UserWarning,
                    stacklevel=2,
                )

        all_errors.extend(row_errors)
        if not row_errors:
            specs[parameter] = DistributionSpec(
                parameter=parameter,
                distribution=distribution,
                spread_lower=spread_lower,
                spread_upper=spread_upper,
                min_abs=min_abs,
            )

    if all_errors:
        raise ValueError(
            "Errors in distributions file:\n"
            + "\n".join(f"  - {e}" for e in all_errors)
        )

    return specs
