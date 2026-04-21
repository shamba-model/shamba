#!/usr/bin/python
"""
### TERMS AND CONDITIONS ###
This software is provided under the University of Edinburgh's Open Technology By
downloading this software you accept the University of Edinburgh's Open Technology
terms and conditions.

These can be viewed here:
https://files.edinburgh-innovations.ed.ac.uk/ei-web/production/images/Small-holder-agriculture-mitigation-benefit-assessment-tool_Terms-and-Conditions-EI.pdf
"""

import csv
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from model.common import csv_handler, io_handler, data_handler

import model.climate as Climate
import model.crop_model as CropModel
import model.crop_params as CropParams
import model.emit as Emit
import model.soil_params as SoilParams
import model.tree_growth as TreeGrowth
import model.tree_model as TreeModel
from model import configuration
from model.common.calculate_emissions import get_location, handle_intervention
import model.common.constants as CONSTANTS

import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
from model.monte_carlo import distribution_handler
from model.monte_carlo.runner import run_monte_carlo, summarise_mc_results, write_mc_summary_csv

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_dir))


def print_crop_emissions(
    crop_base_emissions: np.ndarray,
    crop_project_emissions: np.ndarray,
    crop_difference_emissions: np.ndarray,
):
    table_data = [
        (year, base, proj, proj - base)
        for year, base, proj in zip(
            range(1, len(crop_base_emissions) + 1),
            crop_base_emissions,
            crop_project_emissions,
        )
    ]

    headers = ["Year", "Baseline Emissions", "Projected Emissions", "Difference"]
    table_title = "CROP EMISSIONS (t CO2)"

    print()  # Newline
    print()  # Newline
    print(table_title)
    print("=" * len(table_title))
    print(
        tabulate(
            table_data,
            headers=headers,
            floatfmt=".9f",
            numalign="center",
            tablefmt="fancy_grid",
        )
    )

    print()  # Newline
    print("Total crop difference: ", sum(crop_difference_emissions), " t CO2 ha^-1")
    print("Average crop difference: ", np.mean(crop_difference_emissions))


def print_emissions_table(
    base_emissions, project_emissions, difference, n_years, title
):
    """
    Print a tabular representation of emissions data.

    Args:
        base_emissions (list): List of baseline emissions values.
        project_emissions (list): List of projected emissions values.
        difference (list): List of emission differences.
        n_years (int): Number of years.
        title (str): Title of the emissions table.
    """
    table_data = [
        [
            i + 1,
            f"{base_emissions[i]:.2f}",
            f"{project_emissions[i]:.2f}",
            f"{difference[i]:.2f}",
        ]
        for i in range(n_years)
    ]

    headers = ["Year", "Baseline Emissions", "Projected Emissions", "Difference"]

    print()  # Newline
    print()  # Newline
    print(title)
    print("=" * len(title))
    print(
        tabulate(
            table_data,
            headers=headers,
            floatfmt=".9f",
            numalign="center",
            tablefmt="fancy_grid",
        )
    )
    print()  # Newline
    print(f"Total difference: {sum(difference):.2f}")
    print(f"Average difference: {np.mean(difference):.2f}")


def print_fire_emissions(
    fire_base_emissions, fire_project_emissions, fire_difference, n_years
):
    print_emissions_table(
        fire_base_emissions,
        fire_project_emissions,
        fire_difference,
        n_years,
        "FIRE EMISSIONS (t CO2)",
    )


def print_fertilizer_emissions(
    fertiliser_base_emissions,
    fertiliser_project_emissions,
    fertiliser_difference,
    n_years,
):
    print_emissions_table(
        fertiliser_base_emissions,
        fertiliser_project_emissions,
        fertiliser_difference,
        n_years,
        "FERTILISER EMISSIONS (t CO2)",
    )


def print_litter_emissions(
    litter_base_emissions, litter_project_emissions, litter_difference, n_years
):
    print_emissions_table(
        litter_base_emissions,
        litter_project_emissions,
        litter_difference,
        n_years,
        "LITTER EMISSIONS (t CO2)",
    )


def print_tree_emissions(
    tree_base_emissions, tree_project_emissions, tree_difference, n_years
):
    print_emissions_table(
        tree_base_emissions,
        tree_project_emissions,
        tree_difference,
        n_years,
        "TREE EMISSIONS (t CO2)",
    )


def print_soil_emissions(
    soil_base_emissions, soil_project_emissions, soil_difference, n_years
):
    print_emissions_table(
        soil_base_emissions,
        soil_project_emissions,
        soil_difference,
        n_years,
        "SOIL EMISSIONS (t CO2)",
    )


def print_total_emissions(
    emit_base_emissions, emit_project_emissions, emit_difference, n_years
):
    print_emissions_table(
        emit_base_emissions,
        emit_project_emissions,
        emit_difference,
        n_years,
        "TOTAL EMISSIONS (t CO2)",
    )


def print_tree_projects(tree_projects):
    for project in tree_projects:
        TreeModel.print_biomass(project)
        TreeModel.print_balance(project)


def save_tree_projects(tree_projects, plot_name):
    for i in range(len(tree_projects)):
        TreeModel.save(tree_projects[i], plot_name + f"_tree_proj{i + 1}.csv")


def plot_tree_projects(tree_projects, plot_name):
    for project in tree_projects:
        TreeModel.plot_biomass(project, save_name=plot_name + "_biomassPools.png")
        TreeModel.plot_balance(project, save_name=plot_name + "_massBalance.png")


def print_tree_growths(tree_growths):
    for i in range(len(tree_growths)):
        TreeGrowth.print_to_stdout(tree_growths[i], label=f"growth{i+1}")


def save_tree_growths(tree_growths, plot_name):
    for i in range(len(tree_growths)):
        TreeGrowth.save(tree_growths[i], plot_name + f"_growth{i+1}.csv")


def save_crop_data(base_data, project_data, plot_name, model_type):
    for i, (base, project) in enumerate(zip(base_data, project_data), 1):
        base_filename = f"{plot_name}_{model_type}_base_{i}.csv"
        project_filename = f"{plot_name}_{model_type}_proj_{i}.csv"

        if model_type == "crop_model":
            CropModel.save(base, str(base_filename))
            CropModel.save(project, str(project_filename))
        elif model_type == "crop_params":
            CropParams.save(base, str(base_filename))
            CropParams.save(project, str(project_filename))


def write_emissions_csv(configuration, mod_run, n, st, data):
    # Define the output directory and file name
    output_dir = Path(configuration.OUTPUT_DIR + f"_{mod_run}/plot_{n+st}")
    output_file = output_dir / f"plot_{n+st}_emissions_all_pools_per_year.csv"

    # Ensure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the header and data rows
    header = [
        "emit_base_emissions",
        "emit_project_emissions",
        "emit_difference",
        "soil_base",
        "soil_proj",
        "soil_difference",
        "tree_base",
        "tree_proj",
        "tree_difference",
        "fire_base",
        "fire_project",
        "fire_difference",
        "lit_base",
        "lit_proj",
        "litter_difference",
        "fert_base",
        "fert_proj",
        "fertiliser_difference",
        "crop_base",
        "crop_project",
        "crop_difference",
    ]

    # Write the CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(zip(*data.values()))


def setup_project_directory(project_name, arguments):
    """
    Set up a new project directory with the required input files.

    Args:
    project_name (str): The name of the new project directory.

    Returns:
    str: Path to the newly created project directory.
    """

    # New project directory
    project_dir = os.path.join(configuration.PROJECT_DIR, project_name)

    # Input directory within the project directory
    input_dir = os.path.join(project_dir, "input")

    # Create the project and input directories
    os.makedirs(input_dir, exist_ok=True)

    # List of files to copy
    files_to_copy = [
    ]

    optional_files_to_copy = [
        "climate.csv",
        "soil-info.csv",
        "project_allometry.py",
        "crop_params.csv",
        "tree_params.csv",
        "litter_params.csv",
        "biomass_pool_params.csv",
    ]

    if arguments.get("split-input-file-id") is not None:
        prefix = arguments["split-input-file-id"] 
        files_to_copy.append(str(prefix + "_plot_data.csv"))
        files_to_copy.append(str(prefix + "_mgmt_data.csv"))
        files_to_copy.append(str(prefix + "_tree_size_data.csv"))
        if arguments["use-api"] is False:
            files_to_copy.append(str(prefix + "_climate_cover_data.csv"))
    else:
        files_to_copy.append(arguments["input-file-name"])

    # Source directory (using an existing project as an example)
    source_dir = os.path.join(configuration.PROJECT_DIR, arguments["source-directory"])

    # Copy each file
    for file in files_to_copy:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(input_dir, file)
        if os.path.exists(source_file):
            try: 
                shutil.copy2(source_file, dest_file)
                print(f"Copied {file} to {dest_file}")
            except shutil.SameFileError:
                print(f"File {file} already in source directory")
                pass
        else:
            raise ValueError(f"File {file} does not exist. Please add it to the source directory.")

    # Copy each available optional file
    for file in optional_files_to_copy:
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(input_dir, file)
        if os.path.exists(source_file):
            try: 
                shutil.copy2(source_file, dest_file)
                print(f"Copied {file} to {dest_file}")
            except shutil.SameFileError:
                print(f"File {file} already in source directory")
                pass
        else:
            print(f"Warning: Source file {source_file} does not exist, skipping...")


    print(f"Project setup complete. New project directory: {project_dir}")
    return project_dir


def main(n, arguments):
    project_name = arguments["project-name"]

    # Create a new project directory
    setup_project_directory(project_name, arguments)

    # Get soil model
    soil_model_type = arguments["soil-model"]
    ForwardSoilModel = ForwardSoilModule.get_soil_model(soil_model_type)
    InverseSoilModel = InverseSoilModule.get_soil_model(soil_model_type)

    # Setup the project directory constants
    configuration.SAVE_DIR = os.path.join(configuration.PROJECT_DIR, project_name)

    # specifiying input and output files
    configuration.INPUT_DIR = os.path.join(configuration.SAVE_DIR, "input")
    configuration.OUTPUT_DIR = os.path.join(configuration.SAVE_DIR, "output")

    N_COHORTS = arguments["n-cohorts"]

    if arguments.get("input-file-name") is not None:
        file_path = os.path.join(configuration.INPUT_DIR, arguments["input-file-name"])
        scalar_input_data, tree_size_data, mgmt_input_data, cover_data = data_handler.expand_single_row_data_input(file_path)
        N_YEARS = int(np.atleast_1d(scalar_input_data["yrs_proj"])[0])
        vector_input_data = scalar_input_data | mgmt_input_data | tree_size_data | cover_data

    elif arguments.get("split-input-file-id") is not None:
        prefix = arguments["split-input-file-id"]
        scalar_input_data = data_handler.read_and_validate_timeseries_by_header(
            file_path=os.path.join(configuration.INPUT_DIR, f"{prefix}_plot_data.csv"),
            permitted_vector_lengths=[1],
            target_vector_length=1,
        )
        N_YEARS = int(np.atleast_1d(scalar_input_data["yrs_proj"])[0])
        # TODO: thinning and mortality arrays use N_YEARS+1 entries (year 0 included);
        # other management arrays use N_YEARS. N_YEARS+1 is included here to accommodate
        # both. See the same TODO in broadcast_to_length for full context.
        mgmt_input_data = data_handler.read_and_validate_timeseries_by_header(
            file_path=os.path.join(configuration.INPUT_DIR, f"{prefix}_mgmt_data.csv"),
            permitted_vector_lengths=[1, N_YEARS, N_YEARS + 1],
            target_vector_length=N_YEARS,
        )
        tree_size_data = data_handler.read_and_validate_timeseries_by_header(
            file_path=os.path.join(configuration.INPUT_DIR, f"{prefix}_tree_size_data.csv"),
            permitted_vector_lengths=list(range(5, N_YEARS + 1)),
            target_vector_length=None,
        )
        vector_input_data = scalar_input_data | mgmt_input_data | tree_size_data
        # _climate_cover_data.csv always provides base_cover and proj_cover.
        # It may also contain climate data (Temp, Rain, evap/pet) regardless of use_api,
        # since these are used as a fallback if the API is unavailable. The logic assumes
        # that the file either contains all climate data or none.
        climate_cover_data = data_handler.read_and_validate_timeseries_by_header(
            file_path=os.path.join(configuration.INPUT_DIR, f"{prefix}_climate_cover_data.csv"),
            permitted_vector_lengths=[1] + [i * 12 for i in range(1, N_YEARS + 1)],
            target_vector_length=12 * N_YEARS,
        )
        if "Temp" in climate_cover_data:
            climate_cover_data = data_handler.resolve_evap_pet(climate_cover_data)
        vector_input_data = vector_input_data | climate_cover_data

    validation_errors = (
        data_handler.validate_all_grouped_headers(vector_input_data)
        + data_handler.validate_species_data(vector_input_data)
        + data_handler.validate_required_mgmt_keys(vector_input_data)
    )
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    allometric_keys = arguments["allometric-keys"]

    gwp = arguments["gwp"]

    if arguments["n-samples"]:
        n_samples = arguments["n-samples"]
        if arguments["distribution-file-name"]:
            distribution_file_path = os.path.join(configuration.INPUT_DIR, arguments["distribution-file-name"])
            distribution_dict = distribution_handler.load_distributions(distribution_file_path, vector_input_data)
        else:
            distribution_dict = None
        # TODO: tidy the below up so that it isn't a duplicate of the code at the beginning of handle_intervention()
        use_api = arguments["use-api"]
        no_of_years = vector_input_data[CONSTANTS.NO_OF_YEARS_KEY]
        plot_id = vector_input_data["plot_name"] if "plot_name" in vector_input_data else None
        location = get_location(vector_input_data)
        climate_vectors = None
        if "Temp" in vector_input_data:
            climate_vectors = (
                vector_input_data["Temp"],
                vector_input_data["Rain"],
                vector_input_data["evap"],)
        climate = Climate.from_location(location, use_api, climate_vectors=climate_vectors, n_years=no_of_years)
        soil_params = SoilParams.get_soil_params(
            location=location, use_api=use_api, plot_id=plot_id, plot_index=n
        )

        mc_results = run_monte_carlo(
            base_input_dict=vector_input_data,
            soil_params=soil_params,
            climate=climate,
            n_samples=n_samples,
            create_forward_soil_model=ForwardSoilModel.create,
            create_inverse_soil_model=InverseSoilModel.create,
            n_cohorts=N_COHORTS,
            plot_index=n,
            sample_emission_factors=arguments["sample-emission-factors"],
            distribution_dict=distribution_dict,
            allometry=allometric_keys,
            gwp=gwp,
            use_api=arguments["use-api"],
            seed=arguments["seed"],
        )

        st = 1
        output_dir = Path(configuration.OUTPUT_DIR + f"_{mod_run}") / f"plot_{n + st}"
        output_dir.mkdir(parents=True, exist_ok=True)

        mc_summary = summarise_mc_results(mc_results)
        for scenario, label in [
            (mc_summary.base,    "baseline"),
            (mc_summary.project, "project"),
            (mc_summary.diff,    "diff"),
        ]:
            write_mc_summary_csv(scenario, str(output_dir / f"plot_{n + st}_mc_{label}.csv"))

        # TODO: the MC path currently writes only the quantile summary CSV.  The
        # deterministic path (below) also writes validated input CSVs, per-pool
        # emissions CSVs, soil model CSVs, tree/crop data CSVs, and plots.  Decide
        # which of those outputs are meaningful for an MC run and add them here.
        # Candidates: validated input CSVs (same for all samples — write once);
        # soil/climate CSVs from the base run (representative inputs); plots of the
        # emission distribution (e.g. per-year credible interval fan chart).

        # TODO: the MC path returns early here, so N_YEARS is not used below and
        # the `scalar_input_data`/`mgmt_input_data`/`tree_size_data` variables
        # defined during input loading are never saved.  If validated input saving
        # is added to the MC path, reuse that logic rather than duplicating it.

        emit_diffs = [
            r.emit_project_emissions - r.emit_base_emissions for r in mc_results
        ]
        total_diffs = np.array([float(np.sum(d)) for d in emit_diffs])
        print(
            f"\nMonte Carlo complete: {len(mc_results)} samples\n"
            f"  Summaries written to: {output_dir}\n"
            f"  Total emission difference — mean: {total_diffs.mean():.4f} t CO2 ha^-1  "
            f"  std: {total_diffs.std():.4f} t CO2 ha^-1"
        )
        return
    else:
        intervention_emissions = handle_intervention(
            intervention_input=vector_input_data,
            n_cohorts=N_COHORTS,
            plot_index=n,
            allometry=allometric_keys,
            gwp=gwp,
            use_api=arguments["use-api"],
            create_forward_soil_model=ForwardSoilModel.create,
            create_inverse_soil_model=InverseSoilModel.create,
        )

    # ----------
    # Printing to stdout
    # ----------
    if arguments["print-to-stdout"]:
        # Print some stuff?
        Climate.print_to_stdout(intervention_emissions.climate)
        SoilParams.print_to_stdout(intervention_emissions.soil)

        print_tree_growths(intervention_emissions.tree_growths)

        print_tree_projects(intervention_emissions.tree_projects)

        ForwardSoilModule.print_to_stdout(
            intervention_emissions.for_soil, no_of_years=N_YEARS, label="initialisation"
        )
        ForwardSoilModule.print_to_stdout(
            intervention_emissions.base_forward_soil_data,
            no_of_years=N_YEARS,
            label="baseline",
        )
        ForwardSoilModule.print_to_stdout(
            intervention_emissions.project_forward_soil_data,
            no_of_years=N_YEARS,
            label="project",
        )
        # =============================================================================

        # Crop Emissions
        print_crop_emissions(
            intervention_emissions.crop_base_emissions,
            intervention_emissions.crop_project_emissions,
            intervention_emissions.crop_difference,
        )
        # =============================================================================

        # Fertilizer Emissions
        print_fertilizer_emissions(
            fertiliser_base_emissions=intervention_emissions.fertiliser_base_emissions,
            fertiliser_project_emissions=intervention_emissions.fertiliser_project_emissions,
            fertiliser_difference=intervention_emissions.fertiliser_difference,
            n_years=N_YEARS,
        )
        # =============================================================================

        # Litter Emissions
        print_litter_emissions(
            litter_base_emissions=intervention_emissions.litter_base_emissions,
            litter_project_emissions=intervention_emissions.litter_project_emissions,
            litter_difference=intervention_emissions.litter_difference,
            n_years=N_YEARS,
        )
        # =============================================================================

        # Fire Emissions
        print_fire_emissions(
            fire_base_emissions=intervention_emissions.fire_base_emissions,
            fire_project_emissions=intervention_emissions.fire_project_emissions,
            fire_difference=intervention_emissions.fire_difference,
            n_years=N_YEARS,
        )
        # =============================================================================

        # Tree Eemissions
        print_tree_emissions(
            tree_base_emissions=intervention_emissions.tree_base_emissions,
            tree_project_emissions=intervention_emissions.tree_project_emissions,
            tree_difference=intervention_emissions.tree_difference,
            n_years=N_YEARS,
        )
        # =============================================================================

        # Soil Emissions
        print_soil_emissions(
            soil_base_emissions=intervention_emissions.soil_base_emissions,
            soil_project_emissions=intervention_emissions.soil_project_emissions,
            soil_difference=intervention_emissions.soil_difference,
            n_years=N_YEARS,
        )
        # =============================================================================

    # Total Emissions
    emit_difference = (
        intervention_emissions.emit_project_emissions
        - intervention_emissions.emit_base_emissions
    )

    print_total_emissions(
        emit_base_emissions=intervention_emissions.emit_base_emissions,
        emit_project_emissions=intervention_emissions.emit_project_emissions,
        emit_difference=emit_difference,
        n_years=N_YEARS,
    )
    # =============================================================================

    # Summary of GHG pools
    summary_difference_data = [
        ["Difference Type", "Value", "Units"],
        [
            "Total Crop Difference",
            f"{sum(intervention_emissions.crop_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Fertiliser Difference",
            f"{sum(intervention_emissions.fertiliser_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Litter Difference",
            f"{sum(intervention_emissions.litter_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Fire Difference",
            f"{sum(intervention_emissions.fire_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Tree Difference",
            f"{sum(intervention_emissions.tree_difference):.2f}",
            "t CO2 ha^-1",
        ],
        [
            "Total Soil Difference",
            f"{sum(intervention_emissions.soil_difference):.2f}",
            "t CO2 ha^-1",
        ],
        ["Total Difference", f"{sum(emit_difference):.2f}", "t CO2 ha^-1"],
    ]

    accounting_year = N_YEARS

    summary_difference_title = (
        f"SUMMARY OF EMISSIONS for Year {accounting_year} (t CO2)"
    )

    print()  # Newline
    print()  # Newline
    print(summary_difference_title)
    print("=" * len(summary_difference_title))
    print(tabulate(summary_difference_data, tablefmt="fancy_grid"))
    # =============================================================================

    # Save stuff

    # starting plot output number
    st = 1

    dir = configuration.OUTPUT_DIR + "_" + mod_run + "\plot_" + str(n + st)

    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    plot_name = dir + "\plot_" + str(n + st)

    datasets = [
        ("plot", scalar_input_data),
        ("mgmt", mgmt_input_data),
        ("tree_size", tree_size_data),
    ]

    for name, d in datasets:
        cols = list(d.keys())

        arrays = [np.atleast_1d(np.asarray(d[k], dtype=float)) for k in cols]

        # All columns must be the same length
        target_len = max(a.size for a in arrays)
        padded = []
        for a in arrays:
            if a.size < target_len:
                a = np.pad(a, (0, target_len - a.size), constant_values=np.nan)
            padded.append(a)

        data_to_save = np.column_stack(padded)

        out_path = os.path.join(dir, f"validated_{name}_input_data_{st}.csv")
        csv_handler.print_csv(file_out=out_path, array=data_to_save, col_names=cols)

    if arguments.get("split-input-file-id") is not None:
        cols = list(climate_cover_data.keys())
        data_to_save = np.column_stack([np.asarray(climate_cover_data[k], dtype=float) for k in cols])
        csv_handler.print_csv(file_out=os.path.join(dir, f"validated_climate_data_{st}.csv"), array=data_to_save, col_names=cols)

    Climate.save(intervention_emissions.climate, plot_name + "_climate.csv")

    SoilParams.save(intervention_emissions.soil, plot_name + "_soil.csv")

    save_tree_growths(intervention_emissions.tree_growths, plot_name)

    save_tree_projects(intervention_emissions.tree_projects, plot_name=plot_name)

    save_crop_data(
        intervention_emissions.crop_base,
        intervention_emissions.crop_project,
        plot_name,
        "crop_model",
    )
    save_crop_data(
        intervention_emissions.crop_par_base,
        intervention_emissions.crop_par_project,
        plot_name,
        "crop_params",
    )

    InverseSoilModule.save(
        intervention_emissions.inverse_soil_model, plot_name + "_invSoil.csv"
    )
    ForwardSoilModule.save(
        forward_soil_model=intervention_emissions.for_soil,
        no_of_years=N_YEARS,
        file=plot_name + "_forSoil.csv",
    )

    ForwardSoilModule.save(
        forward_soil_model=intervention_emissions.base_forward_soil_data,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_base.csv",
    )
    ForwardSoilModule.save(
        forward_soil_model=intervention_emissions.project_forward_soil_data,
        no_of_years=N_YEARS,
        file=plot_name + "_soil_model_proj.csv",
    )

    Emit.save(
        intervention_emissions.emit_base_emissions,
        intervention_emissions.emit_project_emissions,
        plot_name + "_emit_proj.csv",
    )

    data = {
        "emit_base_emissions": intervention_emissions.emit_base_emissions,
        "emit_project_emissions": intervention_emissions.emit_project_emissions,
        "emit_difference": emit_difference,
        "soil_base_emissions": intervention_emissions.soil_base_emissions,
        "soil_project_emissions": intervention_emissions.soil_project_emissions,
        "soil_difference": intervention_emissions.soil_difference,
        "tree_base_emissions": intervention_emissions.tree_base_emissions,
        "tree_project_emissions": intervention_emissions.tree_project_emissions,
        "tree_difference": intervention_emissions.tree_difference,
        "fire_base_emissions": intervention_emissions.fire_base_emissions,
        "fire_project_emissions": intervention_emissions.fire_project_emissions,
        "fire_difference": intervention_emissions.fire_difference,
        "litter_base_emissions": intervention_emissions.litter_base_emissions,
        "litter_project_emissions": intervention_emissions.litter_project_emissions,
        "litter_difference": intervention_emissions.litter_difference,
        "fertiliser_base_emissions": intervention_emissions.fertiliser_base_emissions,
        "fertiliser_project_emissions": intervention_emissions.fertiliser_project_emissions,
        "fertiliser_difference": intervention_emissions.fertiliser_difference,
        "crop_base_emissions": intervention_emissions.crop_base_emissions,
        "crop_project_emissions": intervention_emissions.crop_project_emissions,
        "crop_difference": intervention_emissions.crop_difference,
    }

    write_emissions_csv(configuration, mod_run, n, st, data)

    # Plot stuff
    plot_tree_projects(intervention_emissions.tree_projects, plot_name)

    plt.close()

    ForwardSoilModule.plot(
        intervention_emissions.for_soil,
        no_of_years=N_YEARS,
        legend_string="initialisation",
    )

    ForwardSoilModule.plot(
        intervention_emissions.base_forward_soil_data,
        no_of_years=N_YEARS,
        legend_string="baseline",
    )

    ForwardSoilModule.plot(
        intervention_emissions.project_forward_soil_data,
        no_of_years=N_YEARS,
        legend_string="project",
        save_name=plot_name + "_soilModel.png",
    )
    plt.close()

    Emit.plot(intervention_emissions.emit_base_emissions, legend_string="baseline")
    Emit.plot(intervention_emissions.emit_project_emissions, legend_string="project")

    plt.savefig(os.path.join(configuration.OUTPUT_DIR, plot_name + "_emissions.png"))
    plt.close()

    Emit.save(
        emit_base_emissions=intervention_emissions.emit_base_emissions,
        emit_proj_emissions=intervention_emissions.emit_project_emissions,
        file=plot_name + "_emissions.csv",
    )

    return (
        sum(intervention_emissions.crop_difference),
        sum(intervention_emissions.fertiliser_difference),
        sum(intervention_emissions.litter_difference),
        sum(intervention_emissions.fire_difference),
        sum(intervention_emissions.tree_difference),
        sum(intervention_emissions.soil_difference),
        sum(emit_difference),
    )


if __name__ == "__main__":
    number_of_rows = 1
    # number_of_rows = number of plots
    # NOTE: as of v1.2, this code is not fully set up to process multiple plots during the same run.
    # This is on a list of intended updates for the future. To run multiple plots, please
    # run the command line script with individual input files for each plot.

    # Get command line arguments
    arguments = io_handler.get_arguments_interactively()

    mod_run = arguments["output-title"]

    emit_output_data = []
    for n in range(number_of_rows):
        emit_output_data.append(main(n, arguments))
