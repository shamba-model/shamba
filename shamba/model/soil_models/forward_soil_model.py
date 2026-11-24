import math
import os

from typing import Literal, Optional, List, Tuple
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

from .. import configuration, emit
from ..common import csv_handler
import model.soil_models.roth_c.forward_roth_c as roth_c
import model.soil_models.example_soil_model.forward_example as example_soil_model
from .soil_model_types import SoilModelType, ForwardSoilModelData


def get_soil_model(soil_model_type: SoilModelType):
    match soil_model_type:
        case SoilModelType.ROTH_C:
            return roth_c
        case SoilModelType.EXAMPLE:
            return example_soil_model
        case _:
            raise ValueError(f"Unknown soil model type: {soil_model_type}")


def print_to_stdout(
    forward_soil_model: ForwardSoilModelData, no_of_years: int, label: str
) -> None:
    """Print data from forward soil model run to stdout using tabulate with a functional approach."""
    table_title = f"FORWARD CALCULATIONS for {label}"

    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    soil_iom = forward_soil_model.soil.iom

    if len(tot_soc) == no_of_years + 1:
        years = np.arange(len(tot_soc), dtype=float)
    else:
        years = (
            np.arange(-len(tot_soc) + 2, 2, dtype=float) - forward_soil_model.Cy0Year
        )
        years[-1] = 0

    table_data = generate_table_data(
        tot_soc, soil_iom, forward_soil_model.inputs, years
    )

    headers = ["Year", "Carbon", "Crop In", "Tree In"]

    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(tabulate(table_data, headers=headers, floatfmt=".3f", tablefmt="fancy_grid"))


def save(
    forward_soil_model: ForwardSoilModelData,
    no_of_years: int,
    file="soil_model_forward.csv",
):
    tot_soc = np.sum(forward_soil_model.SOC, axis=1)
    inputs = np.append(forward_soil_model.inputs, [[0, 0]], axis=0)
    data = np.column_stack(
        (
            tot_soc + forward_soil_model.soil.iom,
            forward_soil_model.SOC,
            np.array(len(tot_soc) * [forward_soil_model.soil.iom]),
            inputs[:, 0],
            inputs[:, 1],
        )
    )
    cols = ["soc", "dpm", "rpm", "bio", "hum", "iom", "crop_in", "tree_in"]
    if len(tot_soc) != no_of_years + 1:  # solve to value
        cols.insert(0, "year")
        x = np.array(list(range(-len(tot_soc) + 2, 2)))
        x = x - forward_soil_model.Cy0Year
        x[-1] = 0
        data = np.column_stack((x, data))
        csv_handler.print_csv(file, data, col_names=cols)
    else:
        csv_handler.print_csv(file, data, col_names=cols, print_years=True)


def plot(
    forward_soil_model: ForwardSoilModelData,
    legend_string,
    no_of_years: int,
    save_name=None,
):
    # Check if there's already an active figure with the title "Soil Carbon"
    # If so, reuse it for multiple plots on the same figure
    current_fig = plt.gcf()
    if (
        plt.get_fignums()
        and hasattr(current_fig, "_suptitle")
        and current_fig._suptitle is not None
        and "Soil Carbon" in str(current_fig._suptitle)
    ):
        # Reuse existing soil carbon figure
        ax = current_fig.gca()

        # Calculate x and y data for plotting
        tot_soc = np.sum(forward_soil_model.SOC, axis=1) # NB: this excludes IOM pool (which is constant)
        if len(tot_soc) == no_of_years + 1: # TODO: this is brittle - what if initialisation run has this length?
            # baseline or project
            x = list(range(len(tot_soc)))
        else:
            # initialisation run is before year 0
            x = np.array(list(range(-len(tot_soc) + 2, 2)))
            x = x - forward_soil_model.Cy0Year
            x[-1] = 0

        ax.plot(x, tot_soc, label=legend_string)
        ax.legend(loc="best")
    else:
        # Create new figure
        fig = plt.figure()
        fig.suptitle("Soil Carbon")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("SOC (t C ha^-1)")
        ax.set_title("Total soil carbon vs time")

        tot_soc = np.sum(forward_soil_model.SOC, axis=1)
        if len(tot_soc) == no_of_years + 1:
            # baseline or project
            x = list(range(len(tot_soc)))
        else:
            # initialisation run is before year 0
            x = np.array(list(range(-len(tot_soc) + 2, 2)))
            x = x - forward_soil_model.Cy0Year
            x[-1] = 0

        ax.plot(x, tot_soc, label=legend_string)
        ax.legend(loc="best")
    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))

def plot_MC(
        soil_summary: np.ndarray,
        legend_string,
        no_of_years: int,
        save_name=None,
        ):

    data = np.asarray(soil_summary, dtype=float)

    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("soil_summary must have columns [mean, q05, q95]")

    mean, q05, q95 = data.T

    # Baseline/project start at year 0; initialisation ends at year 0.
    if len(mean) == no_of_years + 1:
        x = np.arange(len(mean))
    else:
        x = np.arange(1 - len(mean), 1)

    # Check if there's already an active figure with the title "Soil Carbon"
    # If so, reuse it for multiple plots on the same figure
    current_fig = plt.gcf()
    if (
        plt.get_fignums()
        and hasattr(current_fig, "_suptitle")
        and current_fig._suptitle is not None
        and "Soil Carbon" in str(current_fig._suptitle)
        ):
            # Reuse existing soil carbon figure
            ax = current_fig.gca()
    else:
            # Create new figure
            fig = plt.figure()
            fig.suptitle("Soil Carbon")
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("SOC (t C ha^-1)")
            ax.set_title("Total soil carbon vs time")

    line, = ax.plot(x, mean, label=legend_string)
    ax.fill_between(
        x, q05, q95,
        color=line.get_color(),
        alpha=0.5,
    )
    ax.legend(loc="best")

    if save_name is not None:
        plt.savefig(os.path.join(configuration.OUTPUT_DIR, save_name))


def create_row(
    year: float, carbon: float, inputs: Optional[List[float]] = None
) -> List[float]:
    return [year, carbon] + (inputs or [])


def generate_table_data(
    tot_soc: np.ndarray,
    soil_iom: float,
    inputs: List[Tuple[float, float]],
    years: np.ndarray,
) -> List[List[float]]:
    return [
        create_row(year, soc + soil_iom, list(inputs[i]) if i < len(inputs) else [])
        for i, (year, soc) in enumerate(zip(years, tot_soc))
    ]
