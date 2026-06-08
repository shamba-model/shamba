"""
Module containing tree growth information and allometric functions

def ryan      allometric function based on C. Ryan biotropica paper (2010)
"""

from typing import Dict, List, Tuple, Any, Optional
import logging as log
import math
import os
from typing import Dict, Tuple
import importlib

import matplotlib.pyplot as plt
import numpy as np
from marshmallow import Schema, fields, post_load
from .common.validations import validate_positive_or_zero_numerical_list
from scipy import optimize
from tabulate import tabulate
import model.tree_params as TreeParams

from . import configuration
from .common import csv_handler


# Functions to fit to
def hyperbolic_function(x, a, b):  # Eq. 6.3, SHAMBA model description
    return a * (1 - np.exp(-b * x))


def exponential_1param_function(x, a):  # Eq. 6.2, SHAMBA model description
    return (1 + a) ** x - 1


def linear_function(x, a):  # Eq. 6.1, SHAMBA model description
    return a * x


def logistic_function(x, a, b, c):  # Eq. 6.4, SHAMBA model description
    return a / (1 + np.exp(-b * (x - c)))

def exponential_2param_function(x, a, b): # Eq. 6.5, SHAMBA model description NOTE: 1+a used in exponential functions with a constrained as > 0 to generate growth
    return b * (1 + a) ** x

fitting_functions = {
    "exp1": exponential_1param_function,
    "exp2": exponential_2param_function,
    "hyp": hyperbolic_function,
    "lin": linear_function,
    "log": logistic_function,
}


def exponential_1param_function_inverse(fit_params, agb):
    if math.fabs(agb) < 0.00000001:
        x = 0
    else:
        a = fit_params[0]
        if agb <= -1:
            raise ValueError(f"Invalid agb for exponential inverse: agb={agb}") # This is handled in the above line for when abs(agb) < 0.00000001, but this edge handling is left in for completeness.
        x = math.log(agb + 1) / math.log(a + 1)
    return x


def exponential_1param_function_derivative(
    fit_params, agb
):  # Eq. 7.2, SHAMBA model description
    a = fit_params[0]
    x = exponential_1param_function_inverse(fit_params, agb)
    dagb_dx = ((1 + a) ** x) * (math.log(1 + a))
    return dagb_dx

def exponential_2param_function_inverse(fit_params, agb):
    if math.fabs(agb) < 0.00000001:
        x = 0
    else:
        a = fit_params[0]
        b = fit_params[1]
        if agb <= 0 or b <= 0:
            raise ValueError(f"Invalid arguments for exponential inverse: agb={agb}, b={b}") # The agb condition is handled in the above line for when abs(agb) < 0.00000001, but this edge handling is left in for completeness.
        x = math.log(agb / b)/ math.log(a + 1)
    return x

def exponential_2param_function_derivative(
        fit_params, agb
        ): # Eq. 7.5, SHAMBA model description
    a = fit_params[0]
    b = fit_params[1]
    x = exponential_2param_function_inverse(fit_params, agb)
    dabg_dx = (b * (1 + a) ** x) * (math.log(1 + a))
    return dabg_dx

def hyperbolic_function_inverse(fit_params, agb):
    if math.fabs(agb) < 0.00000001:
        x = 0
    else:
        a = fit_params[0]
        b = fit_params[1]
        if agb >= a:
            raise ValueError(f"No solution exists for hyperbolic inverse: agb ({agb}) >= a ({a})")
        else:
            x = (math.log(a) - math.log(a - agb)) / b
    return x


def hyperbolic_function_derivative(
    fit_params, agb
):  # Eq. 7.3, SHAMBA model description
    a = fit_params[0]
    b = fit_params[1]
    x = hyperbolic_function_inverse(fit_params, agb)
    dagb_dx = a * b * np.exp(-b * x)
    return dagb_dx


def linear_function_derivative(fit_params, agb):  # Eq. 7.1, SHAMBA model description
    a = fit_params[0]
    dagb_dx = a
    return dagb_dx


def logistic_function_inverse(fit_params, agb):
    if math.fabs(agb) < 0.00000001:
        x = 0
    else:
        a = fit_params[0]
        b = fit_params[1]
        c = fit_params[2]
        
        # Handle boundary cases
        if agb >= a:
            raise ValueError(f"No solution exists for logistic inverse: agb ({agb}) >= a ({a})")
        elif agb <= 0:
            raise ValueError(f"No solution exists for logistic inverse: agb ({agb}) <= 0)") # This is handled in the above line for when abs(agb) < 0.00000001, but this edge handling is left in for completeness.
        else:
            # Valid range: 0 < agb < a
            x = c + (math.log(agb) - math.log(a - agb)) / b
    
    return x


def logistic_function_derivative(fit_params, agb):  # Eq. 7.4, SHAMBA model description
    x = logistic_function_inverse(fit_params, agb)

    a = fit_params[0]
    b = fit_params[1]
    c = fit_params[2]
    dagb_dx = (a * b * np.exp(-b * (x - c))) / ((np.exp(-b * (x - c)) + 1) ** 2)

    return dagb_dx


derivative_functions = {
    "exp1": exponential_1param_function_derivative,
    "exp2": exponential_2param_function_derivative,
    "hyp": hyperbolic_function_derivative,
    "lin": linear_function_derivative,
    "log": logistic_function_derivative,
}


class FitData(Schema):
    exp1 = fields.List(fields.Float(allow_nan=True), required=True)
    exp2 = fields.List(fields.Float(allow_nan=True), required=True)
    hyp = fields.List(fields.Float(allow_nan=True), required=True)
    lin = fields.List(fields.Float(allow_nan=True), required=True)
    log = fields.List(fields.Float(allow_nan=True), required=True)


class FitMSEData(Schema):
    exp1 = fields.Float(allow_nan=True)
    exp2 = fields.Float(allow_nan=True)
    hyp = fields.Float(allow_nan=True)
    lin = fields.Float(allow_nan=True)
    log = fields.Float(allow_nan=True)


class TreeGrowth:
    """
    Object holding tree growth data.

    Instance variables
    ----------------
    age                 tree age data in years
    all_fit_data        dict holding fit diameter data in cm for all four fits
    all_fit_params      dict holding fitting params for all four fits
    all_mse             dict holding MSE for all four fits
    allometric_key      string with allometric key
    best                string with best fit ('exp1', 'exp2', 'log', 'lin, or 'hyp')
    biomass             vector with biomass data for each year
    fit_data            dict holding fit diameter data in cm for all four fits
    fit_mse             dict holding MSE for all four fits
    fit_params          dict holding fitting params for all four fits
    tree_diameter       vector with tree diameter data for each year
    """

    def __init__(
        self,
        age,
        all_fit_data,
        all_fit_params,
        all_mse,
        allometric_key,
        best,
        biomass,
        fit_data,
        fit_mse,
        fit_params,
        tree_diameter,
    ):
        self.age = age
        self.all_fit_data = all_fit_data
        self.all_fit_params = all_fit_params
        self.all_mse = all_mse
        self.allometric_key = allometric_key
        self.best = best
        self.biomass = biomass
        self.fit_data = fit_data
        self.fit_mse = fit_mse
        self.fit_params = fit_params
        self.tree_diameter = tree_diameter


class TreeGrowthSchema(Schema):
    age = fields.List(
        fields.Float,
        required=True,
        validate=validate_positive_or_zero_numerical_list,)
    all_fit_data = fields.Nested(FitData, required=True)
    all_fit_params = fields.Nested(FitData, required=True)
    all_mse = fields.Nested(FitMSEData, required=True)
    allometric_key = fields.String(required=True)
    best = fields.String(required=True)
    biomass = fields.List(fields.Float(), required=True)
    fit_data = fields.List(fields.Float(), required=True)
    fit_mse = fields.Float(allow_nan=True)
    fit_params = fields.List(fields.Float(), required=True)
    tree_diameter = fields.List(
        fields.Float,
        required=True,
        validate=validate_positive_or_zero_numerical_list,)

    @post_load
    def build(self, data, **kwargs):
        return TreeGrowth(**data)


def get_biomass(tree_diameter, allometric_key, tree_params):
    if allometric_key in allometric:
        allometric_function = allometric[allometric_key]
        return np.array(
            [allometric_function(diameter, tree_params) for diameter in tree_diameter]
        )
    else: # user should have provided different allometry
        try: 
            project_allometry = importlib.import_module('project_allometry')
            project_allometric = project_allometry.allometric
            allometric_function = project_allometric[allometric_key]
            return np.array([allometric_function(diameter, tree_params) for diameter in tree_diameter])
        except ValueError:
        # Handle case where project allometry is not found
            raise ValueError(f"Allometric key {allometric_key} not found")


def create(
    tree_params: List[TreeParams.TreeParamsData],
    growth_params: Dict[str, np.ndarray],
    allom="chave dry"
) -> TreeGrowth:
    """Create a TreeGrowth object from a tree parameters and a growth parameters
    dictionary.

    Args:
        tree_params (dict): A dictionary containing the tree parameters.
        growth_params (dict): A dictionary containing the growth parameters.
        allom (str, optional): The allometric key. Defaults to "chave dry".

    Returns:
        TreeGrowth: A TreeGrowth object.
    """
    tree_diameter = growth_params.get("diam", np.array([]))
    allometric_key = allom.lower()
    biomass = (
        growth_params["biomass"]
        if "biomass" in growth_params
        else get_biomass(tree_diameter, allometric_key, tree_params)
    )
    age = growth_params["age"]

    all_fit_data, all_fit_params, all_mse = fit(age, biomass)

    # Find key with best fit
    best = min(all_mse, key=lambda k: all_mse[k])
    fit_data = all_fit_data[best]
    fit_params = all_fit_params[best]
    fit_mse = all_mse[best]

    params = {
        "age": age,
        "all_fit_data": all_fit_data,
        "all_fit_params": all_fit_params,
        "all_mse": all_mse,
        "allometric_key": allometric_key,
        "best": best,
        "biomass": biomass,
        "fit_data": fit_data,
        "fit_mse": fit_mse,
        "fit_params": fit_params,
        "tree_diameter": tree_diameter,
    }

    schema = TreeGrowthSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree growth: {errors}")

    return schema.load(params)  # type: ignore


def from_csv(
    tree_params: List[TreeParams.TreeParamsData],
    allometric_key: str,
    input_data: Dict[str, Any],
    sp_index: str = "",
):
    """Construct Growth object using data in a csv file.

    When using input_csv from command line, constructing dictionary of
    np.arrays for each new cohort.

    Args:
        tree_params: TreeParams object (holds tree params)
        allometric_key: string with allometric key
        csv_input_data: dictionary with csv input data
        species_prefix: prefix for species-specific columns (e.g., "sp2_")

    Returns:
        tree_growth: TreeGrowth object
    """
    age_key = f"age_sp{sp_index}"
    age = input_data[age_key]
    diam_key = f"diam_sp{sp_index}"
    diam = input_data.get(diam_key, np.array([]))

    biomass_key = f"biomass_sp{sp_index}"
    if biomass_key in input_data:
        params = {"age": age, "diam": np.array([]), "biomass": input_data[biomass_key]}
    else:
        params = {"age": age, "diam": diam}

    growth = create(tree_params, params, allometric_key)

    return growth


def plot(tree_growth, fit=True, save_name=None):
    """Plot growth data and all four fits in a matplotlib figure."""

    fig = plt.figure()
    fig.suptitle("Tree Growth")

    ax = fig.add_subplot(1, 1, 1)

    ax.plot(tree_growth.age, tree_growth.biomass, "o", label="data")

    if fit:
        for curve in tree_growth.all_fit_data:
            ax.plot(
                tree_growth.age,
                tree_growth.all_fit_data[curve],
                "-",
                label="%s fit" % curve,
            )

        ax.legend(loc="best")

    ax.set_title("Tree biomass vs. Age")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Biomass (kg C /ha) ")

    # Shift ticks so points not cut off
    xticks = ax.get_xticks()
    xmin = xticks[0] - 0.5 * (xticks[1] - xticks[0])
    xmax = xticks[-1] + 0.5 * (xticks[-1] - xticks[-2])
    ax.set_xlim(xmin, xmax)

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))


def print_to_stdout(tree_growth, label, fit=True, params=True, mse=True):
    """Print data and fits for tree growth data to stdout using tabulate."""
    # Prepare the data for tabulate
    table_data = [
        [age, f"{diameter:.2f}", f"{biomass:.2f}"]
        for age, diameter, biomass in zip(
            tree_growth.age, tree_growth.tree_diameter, tree_growth.biomass
        )
    ]

    # Define headers
    headers = ["Age (years)", "Diameter (cm)", "Biomass (kg C)"]

    table_title = f"TREE GROWTH Data for {label}"

    # Print the table using tabulate
    print()  # Newline
    print()  # Newline
    print(table_title)
    print("Source: biomass data" if len(tree_growth.tree_diameter) == 0 else f"Allometric: {tree_growth.allometric_key}")
    print("=" * len(table_title))
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    if fit:
        table_data = [
            [f"{data:.2f}", f"{exp1:.2f}", f"{exp2:.2f}", f"{hyp:.2f}", f"{lin:.2f}", f"{log:.2f}"]
            for data, exp1, exp2, hyp, lin, log in zip(
                tree_growth.biomass,
                tree_growth.all_fit_data["exp1"],
                tree_growth.all_fit_data["exp2"],
                tree_growth.all_fit_data["hyp"],
                tree_growth.all_fit_data["lin"],
                tree_growth.all_fit_data["log"],
            )
        ]
        headers = ["Data", "Exp1.", "Exp2." "Hyp.", "Lin.", "Log."]

        print()  # Newline
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    if params and mse:
        table_data = [
            [
                "MSE",
                f"{tree_growth.all_mse['exp1']:.2f}",
                f"{tree_growth.all_mse['exp2']:.2f}",
                f"{tree_growth.all_mse['hyp']:.2f}",
                f"{tree_growth.all_mse['lin']:.2f}",
                f"{tree_growth.all_mse['log']:.2f}",
            ],
            [
                "a",
                f"{tree_growth.all_fit_params['exp1'][0]:.2f}",
                f"{tree_growth.all_fit_params['exp2'][0]:.2f}",
                f"{tree_growth.all_fit_params['hyp'][0]:.2f}",
                f"{tree_growth.all_fit_params['lin'][0]:.2f}",
                f"{tree_growth.all_fit_params['log'][0]:.2f}",
            ],
            [
                "b",
                "",
                f"{tree_growth.all_fit_params['exp2'][1]:.2f}",
                f"{tree_growth.all_fit_params['hyp'][1]:.2f}",
                "",
                f"{tree_growth.all_fit_params['log'][1]:.2f}",
            ],
            ["c", "", "", "", f"{tree_growth.all_fit_params['log'][2]:.2f}"],
        ]
        headers = ["", "Exp1.", "Exp2", "Hyp.", "Lin.", "Log."]

        print()  # Newline
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


def save(tree_growth, file="tree_growth.csv"):
    """Save growth stuff to a csv file
    Default path is in OUTPUT_DIR.

    Args:
        tree_growth: TreeGrowth object
        file: name or path to csv file

    """
    # growth data
    diam_for_save = (
        tree_growth.tree_diameter
        if len(tree_growth.tree_diameter) > 0
        else np.full(len(tree_growth.age), np.nan)
    )
    csv_handler.print_csv(
        file,
        np.column_stack(
            (tree_growth.age, diam_for_save, tree_growth.biomass)
        ),
        col_names=["age", "diam", "biomass", "source=biomass data" if len(tree_growth.tree_diameter) == 0 else "allom=" + tree_growth.allometric_key],
    )

    # fitted data - in a list because of the size heterogeneity
    fit_file = file.split(".csv")[0] + "_fit.csv"
    data = np.column_stack(
        (
            tree_growth.biomass,
            tree_growth.all_fit_data["exp1"],
            tree_growth.all_fit_data["exp2"],
            tree_growth.all_fit_data["hyp"],
            tree_growth.all_fit_data["lin"],
            tree_growth.all_fit_data["log"],
        )
    )
    cols = ["data", "exp1", "exp2", "hyp", "lin", "log", "best=" + tree_growth.best]
    csv_handler.print_csv(fit_file, data, col_names=cols)

    # fit parameters
    param_file = file.split(".csv")[0] + "_fit_params.csv"
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    for s in ["exp1", "exp2", "hyp", "lin", "log"]:
        row1.append(tree_growth.all_mse[s])
        row2.append(tree_growth.all_fit_params[s][0])
        try:
            row3.append(tree_growth.all_fit_params[s][1])
        except IndexError:
            row3.append("")
        try:
            row4.append(tree_growth.all_fit_params[s][2])
        except IndexError:
            row4.append("")

    data = [row1, row2, row3, row4]
    cols = ["exp1", "exp2", "hyp", "lin", "log"]
    csv_handler.print_csv(param_file, data, col_names=cols)


# --------------------------------------------------
# Begin methods for allometric equations
# NOTE: RETURNS kg C - WHEN USING WITH CARBON POOLS
# (TREE MODEL), MAKE SURE TO CONVERT TO t C
# --------------------------------------------------

DIAMETER_THRESHOLD = 1e-8


# Return AGB of general allometric of type diam = -c*(agb)^c
#   -> solving for agb gives agb = exp[(ln(diam) - ln(c)) / d]
#   -> agb = exp[a + b*ln(diam)]    since c,d are arbitrary
def calculate_above_ground_biomass(
    allometric_params: List[float],
    diameter: float,
    wood_density: Optional[float] = None,
) -> float:
    """
    Calculate above-ground biomass using a general log-type allometric equation.

    This function applies a polynomial function to the log of the diameter,
    then exponentiates the result. If wood density is provided, the result
    is multiplied by the density.

    Args:
        allometric_params: List of coefficients for the allometric equation.
            The first coefficient is multiplied by the highest power of ln(diameter).
            E.g., [2.601, -3.629] represents the equation:
            agb = exp(2.601 * log(diameter) - 3.629)
        diameter: Tree diameter in consistent units (e.g., meters)
        wood_density: Wood density in g/cm^3 (optional, used in some allometric equations)

    Returns:
        float: Above-ground biomass in kg. Returns 0 for non-positive diameters.
    """
    if diameter <= DIAMETER_THRESHOLD:
        return 0.0

    log_diameter = math.log(diameter)
    log_biomass = np.polyval(allometric_params, log_diameter)
    biomass = math.exp(log_biomass)

    if wood_density is not None:
        biomass *= wood_density

    return biomass


# Some specific log allometrics
# All take dbh vectors and tree object as arguments
def ryan(dbh, tree_params):
    # Note: this equation returns AGB in kg C, so no multiplication by tree_params.carbon is needed here
    """C. Ryan, biotropica (2010)."""
    return calculate_above_ground_biomass([2.601, -3.629], dbh)


def tumwebaze_grevillea(dbh, tree_params):
    """Tumwebaze et al. (2013) - Grevillea."""

    agb = calculate_above_ground_biomass(
        [3.06, -5.5], dbh
    ) + calculate_above_ground_biomass([1.32, 1.06], dbh)
    return agb * tree_params.carbon


def tumwebaze_maesopsis(dbh, tree_params):
    """Tumwebaze et al. (2013) - Maesopsis."""

    agb = calculate_above_ground_biomass(
        [3.33, -7.02], dbh
    ) + calculate_above_ground_biomass([2.38, -2.9], dbh)
    return agb * tree_params.carbon


def tumwebaze_markhamia(dbh, tree_params):
    """Tumwebaze et al. (2013) - Markhamia."""
    agb = calculate_above_ground_biomass(
        [2.63, -4.91], dbh
    ) + calculate_above_ground_biomass([2.43, -3.08], dbh)
    return agb * tree_params.carbon


def chave_dry(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with < 1500 mm/year rainfall, > 5 months dry season

    """
    agb = calculate_above_ground_biomass(
        [-0.0281, 0.207, 1.784, -0.730], dbh, wood_density=tree_params.wood_dens
    )
    return agb * tree_params.carbon


def chave_moist(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with 1500-3000 mm/year rainfall, 1-4 months dry season

    """
    agb = calculate_above_ground_biomass(
        [-0.0281, 0.207, 2.148, -1.562], dbh, wood_density=tree_params.wood_dens
    )
    return agb * tree_params.carbon


def chave_wet(dbh, tree_params):
    """Chave et al. (2005) generic tropical tree
    with > 3500 mm/year rainfall, no seasonality

    """
    agb = calculate_above_ground_biomass(
        [-0.0281, 0.207, 1.98, -1.302], dbh, wood_density=tree_params.wood_dens
    )
    return agb * tree_params.carbon


allometric = {
    "ryan": ryan,
    "grevillea": tumwebaze_grevillea,
    "maesopsis": tumwebaze_maesopsis,
    "markhamia": tumwebaze_markhamia,
    "chave dry": chave_dry,
    "chave moist": chave_moist,
    "chave wet": chave_wet,
}


def get_growth(csv_input_data, spp_key, tree_params, allometric_key):
    """Return a Growth object for a single cohort.

    spp_key is a cohort header (e.g. 'proj_species1'). Its value is the
    species code, which is used to look up the corresponding age/diameter
    data (age_sp{code}, diam_sp{code}).
    """
    spp_number = int(np.atleast_1d(csv_input_data[spp_key])[0])
    return from_csv(
        tree_params=tree_params,
        allometric_key=allometric_key,
        input_data=csv_input_data,
        sp_index=spp_number,
    )


def create_baseline_tree_growths(csv_input_data, tree_params, allometric_keys, cohort_count):
    """Return a list of Growth objects for baseline cohorts.
    Baseline allometry always starts at index 0 of allometric_keys.
    """
    return [
        get_growth(
            csv_input_data,
            f"base_species{i + 1}",
            tree_params[i],
            allometric_key=allometric_keys[i],
        )
        for i in range(cohort_count)
    ]


def create_tree_growths(csv_input_data, tree_params, allometric_keys, cohort_count):
    """Return a list of Growth objects for project cohorts.

    Project allometry starts at index (len(allometric_keys) - cohort_count), 
    after baseline allometry keys.
    """
    return [
        get_growth(
            csv_input_data,
            f"proj_species{i + 1}",
            tree_params[i],
            allometric_key=allometric_keys[len(allometric_keys) - cohort_count + i],
        )
        for i in range(cohort_count)
    ]


def fit(
    age: np.ndarray, biomass: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Fit tree growth data using four fitting functions.

    Args:
        age: Array of tree ages
        biomass: Array of corresponding biomass values

    Returns:
        Tuple containing:
        - data: Dict with data points from each of the fits
        - params: Dict with fitting parameters
        - mse: Dict with mean-square error for each fit
    """
    curve_configs = {
        "exp1": {"init": [1], "num_params": 1, "bounds": ([0.0001], [10])},
        "exp2": {"init": [0.05, 0.05], "num_params":2, "bounds": ([0.0001, 0.0001], [10, 1000])},
        "hyp": {"init": [1000, 0.05], "num_params": 2, "bounds": ([0.0001, 0.0001], [100000, 10])},
        "lin": {"init": [1], "num_params": 1, "bounds": ([0.0001], [1000])},
        "log": {"init": [100, 0, 0], "num_params": 3, "bounds": ([0.0001, -100, -100], [100000, 100, 100])},
    }

    data, params, mse = {}, {}, {}

    for curve, config in curve_configs.items():
        try:
            fit_params = optimize.curve_fit(
                fitting_functions[curve], 
                age, 
                biomass, 
                p0=config["init"],
                bounds=config["bounds"],
                maxfev=50000,
                full_output=False
            )
            
            params[curve] = fit_params[0]
            data[curve] = fitting_functions[curve](age, *params[curve])
            mse[curve] = mse_fn(biomass, data[curve])
        except RuntimeError:
            log.warning(f"Could not fit data to {curve}")
            params[curve] = np.array([np.nan] * config["num_params"])
            data[curve] = np.array([np.nan] * len(age))
            mse[curve] = np.inf

    return data, params, mse


def mse_fn(y_mean: np.ndarray, y_real: np.ndarray) -> float:
    """Calculate mean squared error."""
    return float(np.mean((y_mean - y_real) ** 2))
