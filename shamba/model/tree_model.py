#!/usr/bin/python

"""Module containing Tree class."""

import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from marshmallow import Schema, fields, post_load

from . import configuration
from .common import csv_handler
from .common_schema import OutputSchema as ClimateDataOutputSchema
from .tree_growth import TreeGrowthSchema, fitting_functions, derivative_functions
from .tree_params import TreeParamsSchema
from .common.validations import validate_between_0_and_1
import model.common.constants as CONSTANTS

WOODY_AGB_POOLS = [1, 2]  # branch and stem
DEPENDENT_POOLS = [0, 3, 4]  # leaf, coarse root, fine root: determined by woody biomass, not woody NPP


class MassBalanceData(Schema):
    in_ = fields.List(fields.Float(allow_nan=True), required=True)
    out = fields.List(fields.Float(allow_nan=True), required=True)
    acc = fields.List(fields.Float(allow_nan=True), required=True)
    bal = fields.List(fields.Float(allow_nan=True), required=True)


class TreeModel:
    """
    Object for tree model.

    Instance variables
    ----------------
    tree_params     TreeParams object with the params (wood_dens, carbon, etc.)
    tree_growth     TreeGrowth object governing growth of trees
    alloc           vector with allocation for each year
    turnover        vector with turnover for each year
    thinning_fraction       vector with thinning fraction left for each year
    mortality_fraction       vector with mortality fraction left for each year
    thinning            vector with thinning regime for each year
    mortality            vector with mortality regime for each year
    output          output to soil,fire in t C ha^-1
                    (dict with keys 'carbon,'nitrogen','DMon','DMoff')
    stand_biomass   vector with yearly woody biomass pools
    balance         MassBalanceData object with mass balance data
    """

    def __init__(
        self,
        tree_params,
        tree_growth,
        alloc,
        turnover,
        thinning_fraction,
        mortality_fraction,
        thinning,
        mortality,
        output,
        stand_biomass,
        balance,
    ):
        self.tree_params = tree_params
        self.tree_growth = tree_growth
        self.alloc = alloc
        self.turnover = turnover
        self.thinning_fraction = thinning_fraction
        self.mortality_fraction = mortality_fraction
        self.thinning = thinning
        self.mortality = mortality
        self.output = output
        self.stand_biomass = np.array(stand_biomass)
        self.balance = balance


class TreeModelSchema(Schema):
    tree_params = fields.Nested(TreeParamsSchema, required=True)
    tree_growth = fields.Nested(TreeGrowthSchema, required=True)
    alloc = fields.List(fields.Float, required=True)
    turnover = fields.List(
        fields.Float,
        required=True,
        validate=validate_between_0_and_1,
    )

    thinning_fraction = fields.List(
        fields.Float,
        required=True,
        validate=validate_between_0_and_1,
    )

    mortality_fraction = fields.List(
        fields.Float,
        required=True,
        validate=validate_between_0_and_1,
    )

    thinning = fields.List(
        fields.Float,
        required=True,
        validate=validate_between_0_and_1,
    )

    mortality = fields.List(
        fields.Float,
        required=True,
        validate=validate_between_0_and_1,
    )
    output = fields.Nested(ClimateDataOutputSchema, required=True)
    stand_biomass = fields.List(fields.List(fields.Float), required=True)
    balance = fields.Nested(MassBalanceData, required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeModel(**data)


def create(
    tree_params,
    tree_growth,
    pool_params,
    no_of_years,
    initial_stand_density,
    year_planted=0,
    thinning=None,
    mortality=None,
) -> TreeModel:
    """Intialise TreeModel object (run biomass model, essentially).

    Args:
        tree_params         TreeParams object (holds tree params)
        tree_growth         TreeGrowth object
        pool_params         dict with alloc, turnover, thinning_fraction, and mortality_fraction
        year_planted         year (after start of project) tree is planted
        initial_stand_density    initial stand density
        thinning                vector with thinning regime for each year
        mortality                vector with mortality regime for each year

    Returns:
        tree_model: TreeModel object
    """
    alloc = pool_params["alloc"]
    turnover = pool_params["turnover"]
    thinning_fraction = pool_params["thinning_fraction"]
    mortality_fraction = pool_params["mortality_fraction"]
    thinning = np.zeros(no_of_years + 1) if thinning is None else thinning
    mortality = np.zeros(no_of_years + 1) if mortality is None else mortality
    # Set initial_WAGB_tree to biomass at age (x) = 1 year. The fitting functions expect the age followed by parameter values (* unpacks fit_params).
    initial_WAGB_tree = max(fitting_functions[tree_growth.best](1, *tree_growth.fit_params), 0)

    output, stand_biomass, balance = get_inputs(
        tree_params=tree_params,
        tree_growth=tree_growth,
        alloc=alloc,
        turnover=turnover,
        thinning_fraction=thinning_fraction,
        mortality_fraction=mortality_fraction,
        initial_WAGB_tree=initial_WAGB_tree,
        year_planted=year_planted,
        initial_stand_dens=initial_stand_density,
        thinning=thinning,
        mortality=mortality,
        no_of_years=no_of_years,
    )

    params = {
        "tree_params": vars(tree_params),
        "tree_growth": vars(tree_growth),
        "alloc": alloc,
        "turnover": turnover,
        "thinning_fraction": thinning_fraction,
        "mortality_fraction": mortality_fraction,
        "thinning": thinning,
        "mortality": mortality,
        "output": output,
        "stand_biomass": stand_biomass,
        "balance": balance,
    }

    schema = TreeModelSchema()
    errors = schema.validate(params)

    if errors != {}:
        print(f"Errors in tree model: {errors}")

    return schema.load(params)  # type: ignore


def from_defaults(
    tree_params,
    tree_growth,
    no_of_years,
    stand_density,
    year_planted=0,
    thinning=None,
    thinning_fraction_woody=None,
    mortality=None,
    mortality_fraction_woody=None,
):
    """Use defaults for pool params.

    Thinning/mortality fractions for leaf, coarse root and fine root
    (DEPENDENT_POOLS) always come from biomass_pool_params.csv. Only the
    branch/stem (WOODY_AGB_POOLS) fractions can be overridden, since those
    come from the per-project intervention input.
    """

    data = csv_handler.read_csv("biomass_pool_params.csv", cols=(1, 2, 3, 4))
    turnover = data[:, 0]
    alloc = data[:, 1]
    if abs(alloc[1:3].sum() - 1) > 1e-8:
        raise ValueError(
            "WAGB allocation fractions in biomass_pool_params.csv do not sum to 1."
            " Please check the file and correct the allocation fractions for branch and stem."
        )

    thinning_fraction = np.array(data[:, 2], copy=True)
    mortality_fraction = np.array(data[:, 3], copy=True)

    # Take into account croot alloc - rs * stem alloc
    alloc[3] = alloc[2] * tree_params.root_to_shoot

    # thinning and mortality
    if thinning is None:
        thinning = np.zeros(no_of_years + 1)
    if thinning_fraction_woody is not None:
        thinning_fraction[WOODY_AGB_POOLS] = thinning_fraction_woody
    if mortality is None:
        mortality = np.zeros(no_of_years + 1)
    if mortality_fraction_woody is not None:
        mortality_fraction[WOODY_AGB_POOLS] = mortality_fraction_woody

    params = {
        "alloc": alloc,
        "turnover": turnover,
        "thinning_fraction": thinning_fraction,
        "mortality_fraction": mortality_fraction,
    }
    return create(
        tree_params=tree_params,
        tree_growth=tree_growth,
        pool_params=params,
        no_of_years=no_of_years,
        year_planted=year_planted,
        initial_stand_density=stand_density,
        thinning=thinning,
        mortality=mortality,
    )

def calculate_fluxes(flux, pools, input_params, year_index):
    """Compute dead/thinning/live biomass fluxes for the given year.
        Ensures that total fluxes cannot be more than 1 x pool size."""
    remaining_biomass = np.array(pools, copy = True)
    
    flux["thinning"][year_index] = remaining_biomass * input_params["thinning"][year_index]
    remaining_biomass -= flux["thinning"][year_index]

    flux["dead"][year_index] = remaining_biomass * input_params["dead"][year_index]
    remaining_biomass -= flux["dead"][year_index]

    flux["live"][year_index] = remaining_biomass * input_params["live"][year_index]
    
    return flux


def get_inputs(
    thinning,
    mortality,
    turnover,
    thinning_fraction,
    mortality_fraction,
    alloc,
    tree_growth,
    tree_params,
    initial_WAGB_tree,
    year_planted,
    initial_stand_dens,
    no_of_years,
):
    """
    Calculate and return residues and soil inputs from the tree.

    Args:
        same explanations as create
    Returns:
        output: dict with soil,fire inputs due to this tree
                        (keys='carbon','nitrogen','DMon','DMoff')
        stand_biomass: vector with yearly woody biomass pools

    **NOTE** a lot of these arrays are implicitly 2d with 2nd
    dimension = [leaf, branch, stem, croot, froot]. Careful here.
    """
    # NOTE - initially quantities are in kg C
    #   -> stand_biomass and output get converted at end before returning

    # Get params
    stand_density = np.zeros(no_of_years + 1)
    stand_density[year_planted] = initial_stand_dens

    input_params = {
        "live": np.array((no_of_years + 1) * [turnover]),
        "thinning": thinning,
        "dead": mortality,
    }
    retained_fraction = {
        "live": 1,
        "thinning": thinning_fraction,
        "dead": mortality_fraction,
    }

    # initialise stuff
    tree_pools = np.zeros((no_of_years + 1, 5))
    stand_biomass = np.zeros((no_of_years + 1, 5))
    t_NPP = np.zeros(no_of_years + 1)

    flux = {}
    inputs = {}
    exports = {}
    for s in ["live", "dead", "thinning"]:
        flux[s] = np.zeros((no_of_years + 1, 5))
        inputs[s] = np.zeros((no_of_years + 1, 5))
        exports[s] = np.zeros((no_of_years + 1, 5))

    input_carbon = np.zeros((no_of_years + 1, 5))
    export_carbon = np.zeros((no_of_years + 1, 5))
    biomass_growth = np.zeros((no_of_years + 1, 5))

    # set stand_biomass[0] to initial (allocated appropriately)
    tree_pools[year_planted] = initial_WAGB_tree * alloc # initial_WAGB_tree = initial woody AGB
    stand_biomass[year_planted] = tree_pools[year_planted] * stand_density[year_planted]

    in_ = np.zeros(no_of_years + 1)
    acc = np.zeros(no_of_years + 1)
    bal = np.zeros(no_of_years + 1)
    out = np.zeros(no_of_years + 1)

    for i in range(1 + year_planted, no_of_years + 1):
        # Careful with indices - using 1-based here
        #   since, e.g., stand_biomass[2] should correspond to
        #   biomass after 2 years

        wagb_tree = sum(tree_pools[i - 1][WOODY_AGB_POOLS])

        # Growth for one tree
        t_NPP[i] = derivative_functions[tree_growth.best](tree_growth.fit_params, wagb_tree)
        biomass_growth[i][WOODY_AGB_POOLS] = t_NPP[i] * alloc[WOODY_AGB_POOLS] * stand_density[i - 1]

        stand_biomass[i][WOODY_AGB_POOLS] = stand_biomass[i - 1][WOODY_AGB_POOLS]
        stand_biomass[i][WOODY_AGB_POOLS] += biomass_growth[i][WOODY_AGB_POOLS]
        
        WAGB_biomass_total = sum(stand_biomass[i][WOODY_AGB_POOLS])
        stand_biomass[i][DEPENDENT_POOLS] = WAGB_biomass_total*alloc[DEPENDENT_POOLS]

        flux = calculate_fluxes(
            flux=flux,
            pools=stand_biomass[i],
            input_params=input_params,
            year_index=i
        )

        stand_biomass[i] -= sum(flux.values())[i] # this applies mortality, turnover and thinning
        biomass_growth[i][DEPENDENT_POOLS] = stand_biomass[i][DEPENDENT_POOLS] - stand_biomass[i-1][DEPENDENT_POOLS] + sum(flux.values())[i][DEPENDENT_POOLS]
        
        for s in inputs:
            inputs[s][i] = retained_fraction[s] * flux[s][i]
            exports[s][i] = (1 - retained_fraction[s]) * flux[s][i]

        # Totals (in t C / ha)
        input_carbon[i] = sum(inputs.values())[i]
        export_carbon[i] = sum(exports.values())[i]

        stand_density[i] = stand_density[i - 1]
        stand_density[i] *= 1 - (input_params["thinning"][i])
        stand_density[i] *= 1 - (input_params["dead"][i])
        if stand_density[i] < 1:
            print("SD [i] is less than 1, end of this tree cohort...")
            break
        tree_pools[i] = stand_biomass[i] / stand_density[i]

        # Balance stuff

        in_[i] = biomass_growth[i].sum()
        acc[i] = stand_biomass[i].sum() - stand_biomass[i - 1].sum()
        out[i] = input_carbon[i].sum() + export_carbon[i].sum()
        bal[i] = in_[i] - out[i] - acc[i]

    # *********************
    # Standard output stuff
    # in tonnes
    # *********************
    mass_balance = {
        "in_": in_ * 0.001,
        "out": out * 0.001,
        "acc": acc * 0.001,
        "bal": bal * 0.001,
    }
    stand_biomass *= 0.001  # convert to tonnes for emissions calc.
    C = input_carbon[0:no_of_years]
    DM = C / tree_params.carbon
    N = np.zeros((no_of_years, 5))
    for i in range(no_of_years):
        N[i] = tree_params.nitrogen * DM[i]

    output = {}
    output["above"] = {
        "carbon": 0.001 * (C[:, 0] + C[:, 1] + C[:, 2]),
        "nitrogen": 0.001 * (N[:, 0] + N[:, 1] + N[:, 2]),
        "DMon": 0.001 * (DM[:, 0] + DM[:, 1] + DM[:, 2]),
        "DMoff": np.zeros(len(C[:, 0])),
    }
    output["below"] = {
        "carbon": 0.001 * CONSTANTS.TREE_ROOT_IN_TOP_30 * (C[:, 3] + C[:, 4]),
        "nitrogen": 0.001 * CONSTANTS.TREE_ROOT_IN_TOP_30 * (N[:, 3] + N[:, 4]),
        "DMon": 0.001 * CONSTANTS.TREE_ROOT_IN_TOP_30 * (DM[:, 3] + DM[:, 4]),
        "DMoff": np.zeros(len(C[:, 0])),
    }
    
    return output, stand_biomass, mass_balance


def plot_biomass(tree_model, save_name=None):
    """Plot the biomass pool data."""

    fig = plt.figure()
    fig.suptitle("Biomass Pools")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tree_model.stand_biomass[:, 0], label="leaf")
    ax.plot(tree_model.stand_biomass[:, 1], label="branch")
    ax.plot(tree_model.stand_biomass[:, 2], label="stem")
    ax.plot(tree_model.stand_biomass[:, 3], label="coarse root")
    ax.plot(tree_model.stand_biomass[:, 4], label="fine root")
    ax.legend(loc="best")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Pool biomass (t C ha^-1)")
    ax.set_title("Biomass pools vs time")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))

    plt.close()


def plot_balance(tree_model, save_name=None):
    """Plot the mass balance data."""

    fig = plt.figure()
    fig.suptitle("Biomass Mass Balance")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tree_model.balance["in_"], label="in")
    ax.plot(tree_model.balance["out"], label="out")
    ax.plot(tree_model.balance["acc"], label="accumulated")
    ax.plot(tree_model.balance["bal"], label="balance")
    ax.legend(loc="best")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Biomass (t C ha^-1)")
    ax.set_title("Mass balance vs time")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))

    plt.close()


def print_biomass(tree_model):
    total_biomass = np.sum(tree_model.stand_biomass, axis=1)

    # Prepare the data for tabulate
    table_data = [
        [year, f"{biomass:.2f}"]  # Format biomass to 2 decimal places
        for year, biomass in enumerate(total_biomass)
    ]

    # Define headers
    headers = ["Year", "Biomass"]
    table_title = "BIOMASS MODEL"

    # Print the table using tabulate
    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(
        tabulate(table_data, headers=headers, numalign="center", tablefmt="fancy_grid")
    )


def print_balance(tree_model):
    print("\nMass-balance sum (kg C /ha): ", np.sum(tree_model.balance["bal"]))
    to_difference = np.sum(tree_model.balance["bal"]) / np.sum(
        tree_model.stand_biomass[-1]
    )
    print("Normalized mass balance (kg C /ha): ", to_difference)


def save(tree_model, file="tree_model.csv"):
    """Save output and biomass to a csv file.
    Default path is in OUTPUT_DIR.

    Args:
        file: name or path to csv

    """
    # outputs
    cols = []
    data = []
    for s1 in ["above", "below"]:
        for s2 in ["carbon", "nitrogen", "DMon", "DMoff"]:
            cols.append(s2 + "_" + s1)
            data.append(tree_model.output[s1][s2])
    data = np.column_stack(tuple(data))
    csv_handler.print_csv(file, data, col_names=cols)

    # biomass
    biomass_file = file.split(".csv")[0] + "_biomass.csv"
    cols = ["leaf", "branch", "stem", "croot", "froot"]
    csv_handler.print_csv(
        biomass_file,
        tree_model.stand_biomass,
        col_names=cols,
        print_total=True,
        print_years=True,
    )


def create_tree_projects(
    csv_input_data,
    tree_params,
    growths,
    thinning_project,
    thinning_fraction_woody_project,
    mortality_project,
    mortality_fraction_woody_project,
    no_of_years,
    cohort_count,
):
    return [
        from_defaults(
            tree_params=tree_params[i],
            tree_growth=growths[i],
            year_planted=int(csv_input_data[f"proj_plant_yr{i + 1}"]),
            stand_density=int(csv_input_data[f"proj_plant_dens{i + 1}"]),
            thinning=thinning_project,
            thinning_fraction_woody=thinning_fraction_woody_project,
            mortality=mortality_project,
            mortality_fraction_woody=mortality_fraction_woody_project,
            no_of_years=no_of_years,
        )
        for i in range(cohort_count)
    ]
