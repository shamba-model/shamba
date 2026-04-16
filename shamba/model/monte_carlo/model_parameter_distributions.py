"""IPCC emission factor distributions set and helper functions."""

from model.monte_carlo.distribution_handler import DistributionSpec
from model.common.constants import (
    ef_burn_default,
    ef_N_inputs_default,
    combustion_factor_default,
    volatile_frac_organic_fertiliser_default,
    volatile_frac_synthetic_fertiliser_default,
)
from math import log as ln

RELATIVE_EF_BURN_STD = 0.3  # relative standard deviation for ef_burn parameters (30% of the mean). Incomplete data, based on available values.
RELATIVE_CF_STD = 0.4  # relative standard deviation for combustion_factor parameters (40% of the mean). Incomplete data, based on available values.
TRUNCNORMAL_VOL_ORG_SPREAD = ((0.31-volatile_frac_organic_fertiliser_default)/1.96)/volatile_frac_organic_fertiliser_default  # spread for truncnormal distribution of volatile fraction of organic fertiliser, based on available 95% CI [0.31, 0.00], truncated normal copes with 0 lower bound.
LOGNORMAL_VOL_SYN_SPREAD = (ln(0.33/0.02)/(2*1.96))  # spread for lognormal distribution of volatile fraction of synthetic fertiliser, based on available data [0.33, 0.02], with 95% confidence interval. NOTE: this gives a spread ~0.72>0.5 and will result in a warning. TODO: confirm this is the best distribution choice


MODEL_PARAMETER_DISTRIBUTIONS = {
    "ef_burn_crop_N2O": DistributionSpec(parameter="ef_burn_crop_N2O", distribution="normal", spread_lower=RELATIVE_EF_BURN_STD, spread_upper=RELATIVE_EF_BURN_STD, min_abs=None),
    "ef_burn_crop_CH4": DistributionSpec(parameter="ef_burn_crop_CH4", distribution="normal", spread_lower=RELATIVE_EF_BURN_STD, spread_upper=RELATIVE_EF_BURN_STD, min_abs=None),
    "ef_burn_tree_N2O": DistributionSpec(parameter="ef_burn_tree_N2O", distribution="normal", spread_lower=RELATIVE_EF_BURN_STD, spread_upper=RELATIVE_EF_BURN_STD, min_abs=None),
    "ef_burn_tree_CH4": DistributionSpec(parameter="ef_burn_tree_CH4", distribution="normal", spread_lower=RELATIVE_EF_BURN_STD, spread_upper=RELATIVE_EF_BURN_STD, min_abs=None),
    "ef_N_inputs": DistributionSpec(parameter="ef_N_inputs", distribution="normal", spread_lower=0.004/ef_N_inputs_default, spread_upper=0.004/ef_N_inputs_default, min_abs=None),
    "combustion_factor_crop": DistributionSpec(parameter="combustion_factor_crop", distribution="normal", spread_lower=RELATIVE_CF_STD, spread_upper=RELATIVE_CF_STD, min_abs=None),
    "combustion_factor_tree": DistributionSpec(parameter="combustion_factor_tree", distribution="normal", spread_lower=RELATIVE_CF_STD, spread_upper=RELATIVE_CF_STD, min_abs=None),
    "volatile_frac_synthetic_fertiliser": DistributionSpec(
        parameter="volatile_frac_synthetic_fertiliser", distribution="lognormal", spread_lower=LOGNORMAL_VOL_SYN_SPREAD, spread_upper=LOGNORMAL_VOL_SYN_SPREAD, min_abs=None
    ),
    "volatile_frac_organic_fertiliser": DistributionSpec(
        parameter="volatile_frac_organic_fertiliser", distribution="truncated_normal", spread_lower=TRUNCNORMAL_VOL_ORG_SPREAD, spread_upper=TRUNCNORMAL_VOL_ORG_SPREAD, min_abs=None
        ),
    }

def expand_model_param_samples(model_param_sample):
    """Expand a model parameter sample with the parameters needed for the emit function."""
    model_param_sample_expanded = model_param_sample.copy()
    # Add ef_burn parameters
    model_param_sample_expanded["ef_burn"] = {
        param: model_param_sample[f"ef_burn_{param}"]
        for param in ["crop_N2O", "crop_CH4", "tree_N2O", "tree_CH4"]
        }
    # Add combustion_factor parameters
    model_param_sample_expanded["combustion_factor"] = {
        param: model_param_sample_expanded[f"combustion_factor_{param}"]
        for param in ["crop", "tree"]
    }
    return model_param_sample_expanded

def flatten_model_param_sample(model_param_sample):
    """Flatten a model parameter sample by removing the nested structure."""
    flat_sample = model_param_sample.copy()
    # Remove ef_burn nested dict
    for param in ["crop_N2O", "crop_CH4", "tree_N2O", "tree_CH4"]:
        flat_sample[f"ef_burn_{param}"] = flat_sample["ef_burn"][param]
    del flat_sample["ef_burn"]
    # Remove combustion_factor nested dict
    for param in ["crop", "tree"]:
        flat_sample[f"combustion_factor_{param}"] = flat_sample["combustion_factor"][param]
    del flat_sample["combustion_factor"]
    return flat_sample