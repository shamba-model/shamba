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
    return schema.load(params)  # type: ignore


_BIOMASS_POOLS = ("leaf", "branch", "stem", "croot", "froot")


def load_biomass_pool_species_data(
    filename: str = "biomass_pool_params.csv",
) -> dict:
    """Load per-species biomass pool params from csv, keyed by each row's own
    Sc (species code) and pool name columns — not by row position.

    The file must contain exactly one row per (species, pool) combination,
    covering leaf/branch/stem/croot/froot for each species — rows for a
    species may appear in any order.

    Args:
        filename: Name of the CSV file to load.

    Returns:
        Dict mapping species code to a dict with turnover, alloc,
        thinning_fraction, and mortality_fraction arrays (one value per pool,
        in leaf/branch/stem/croot/froot order).
    """
    resolved_path = csv_handler.resolve_csv_path(filename)
    try:
        numeric = np.atleast_2d(
            np.genfromtxt(resolved_path, skip_header=1, usecols=(0, 2, 3, 4, 5), delimiter=",", comments="#")
        )
        pool_names = np.atleast_1d(
            np.genfromtxt(resolved_path, skip_header=1, usecols=(1,), dtype=str, delimiter=",", comments="#")
        )
    except ValueError as e:
        raise ValueError(
            f"'{filename}' could not be read as a biomass pool params file. It must have "
            f"columns 'Sc,pool,TO,AL,THf,DTf' — an older file without the 'Sc' column is "
            f"no longer supported and needs to be migrated. Original error: {e}"
        ) from e

    if np.isnan(numeric).any():
        bad_rows = [r + 2 for r in np.where(np.isnan(numeric).any(axis=1))[0]]
        raise ValueError(
            f"'{filename}' has a missing/blank numeric value in row(s) {bad_rows} "
            f"(counting the header as row 1). Every row must have a value in "
            f"every column (Sc, TO, AL, THf, DTf)."
        )

    rows_by_species: dict = {}
    for i in range(numeric.shape[0]):
        species = int(numeric[i, 0])
        pool = str(pool_names[i]).strip().lower()
        if pool not in _BIOMASS_POOLS:
            raise ValueError(
                f"'{filename}' row {i + 2} has pool name '{pool_names[i]}', "
                f"which is not one of {_BIOMASS_POOLS}."
            )
        species_rows = rows_by_species.setdefault(species, {})
        if pool in species_rows:
            raise ValueError(
                f"'{filename}' has more than one '{pool}' row for species code {species}."
            )
        species_rows[pool] = numeric[i, 1:]

    species_data = {}
    for species, pools in rows_by_species.items():
        missing = [p for p in _BIOMASS_POOLS if p not in pools]
        if missing:
            raise ValueError(
                f"'{filename}' is missing row(s) for pool(s) {missing} for species "
                f"code {species}. Every species needs exactly one row per pool: "
                f"{_BIOMASS_POOLS}."
            )
        ordered = np.array([pools[p] for p in _BIOMASS_POOLS])
        species_data[species] = {
            "turnover": ordered[:, 0],
            "alloc": ordered[:, 1],
            "thinning_fraction": ordered[:, 2],
            "mortality_fraction": ordered[:, 3],
        }
    return species_data


def from_defaults(
    tree_params,
    tree_growth,
    no_of_years,
    stand_density,
    pool_species_data,
    year_planted=0,
    thinning=None,
    thinning_fraction=None,
    mortality=None,
    mortality_fraction=None,
):
    """Use defaults for pool params.
    Can override defaults for thinning_fraction and mortality_fraction by providing arguments.

    Args:
        pool_species_data: pre-loaded biomass pool data (as returned by
            load_biomass_pool_species_data()), loaded once per run by the caller.
    """

    pool_data = pool_species_data
    if tree_params.species not in pool_data:
        raise KeyError(
            f"No biomass pool parameters found for species '{tree_params.species}' "
            "in biomass_pool_params.csv. Add a 5-row block (leaf, branch, stem, "
            "croot, froot) for this species, in the same order as tree_params.csv."
        )
    species_pool_data = pool_data[tree_params.species]
    turnover = species_pool_data["turnover"]
    # pool_data may be a dict shared across every cohort of this species (see
    # pool_species_data above) rather than freshly read each call — copy
    # before any in-place mutation, or it corrupts the value for every other
    # cohort using the same species.
    alloc = species_pool_data["alloc"].copy()
    temp_thinning_fraction = species_pool_data["thinning_fraction"]
    temp_mortality_fraction = species_pool_data["mortality_fraction"]

    # Take into account croot alloc - rs * stem alloc
    alloc[3] = alloc[2] * tree_params.root_to_shoot

    # thinning and mortality
    if thinning is None:
        thinning = np.zeros(no_of_years + 1)
    if thinning_fraction is None:
        thinning_fraction = temp_thinning_fraction
    if mortality is None:
        mortality = np.zeros(no_of_years + 1)
    if mortality_fraction is None:
        mortality_fraction = temp_mortality_fraction

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

    WOODY_AGB_POOLS = [1, 2] # branch and stem
    DEPENDENT_POOLS = [0, 3, 4] # leaf, coarse and fine roots: these pools are determined by the amount of woody biomass - not woody NPP

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
    cols = list(_BIOMASS_POOLS)
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
    thinnings_project,
    thinning_fractions_project,
    mortalities_project,
    mortality_fractions_project,
    no_of_years,
    cohort_count,
    type,
    pool_species_data,
):
    return [
        from_defaults(
            tree_params=tree_params[i],
            tree_growth=growths[i],
            year_planted=int(np.atleast_1d(csv_input_data[f"{type}_plant_yr{i + 1}"])[0]),
            stand_density=int(np.atleast_1d(csv_input_data[f"{type}_plant_dens{i + 1}"])[0]),
            thinning=thinnings_project[i],
            thinning_fraction=thinning_fractions_project[i],
            mortality=mortalities_project[i],
            mortality_fraction=mortality_fractions_project[i],
            no_of_years=no_of_years,
            pool_species_data=pool_species_data,
        )
        for i in range(cohort_count)
    ]
