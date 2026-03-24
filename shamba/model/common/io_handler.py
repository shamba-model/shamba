#!/usr/bin/python

"""Module for io related functions in the SHAMBA program."""

import logging as log
import os
from datetime import datetime
import questionary
from questionary import Validator, ValidationError
from questionary import Choice

from model import configuration
import model.tree_growth as TreeGrowth
from model.soil_models.soil_model_types import SoilModelType
from model.common.constants import (
    DEFAULT_USE_API,
    DEFAULT_ALLOMORPHY,
    DEFAULT_GWP,
    GWP_list,
)
from model.common.validations import validate_integer, validate_numerical


def get_arguments_interactively():
    """
    Prompt the user for arguments interactively using the `questionary` library.
    Return a dictionary containing the argument values.
    """
    arguments = {}

    # Display instructions using a pure print — not necessary to prompt here
    print(
        """
INSTRUCTIONS

___________________________STEP 1: create main input file(s) _____________________
--------------------------- Option 1: Single input file---------------------------
Complete in full the Excel worksheet 'SHAMBA input output template v1.2',
(located in the 'data-input-templates' folder) including all references 
for information. The reviewer will reject the modelling unless it is fully
referenced. See the instructions in the Excel worksheet.

On the '_questionnaire' worksheet, you must enter a value in each of the
blue cells in the 'Input data' column (column K) in response to each 
'data collection question', otherwise the model will not run properly. 
If the question is not relevant to the land use you are modelling, enter zero.

To run the model for a particular intervention, save the `input` sheet from the 
template as a .csv file into a new `shamba/projects/"project-name"/input`
folder. This is the 'source directory'
you must specify when prompted at the command line.

-------------- Option 2: prepare split input files (vector format) ----------------
Instead of a single, one row input csv (Option 1), you can provide data split 
across four csv files.

This allows parameters to vary year-by-year and therefore results in
more accurate modelling of carbon changes and greenhouse gas emissions.
For example: the single-row input file allows a single crop yield value, applied 
over one growth phase (between start year and end year). Split input files allow 
you to enter different crop yields each year - which could represent changing 
seeding rates, a crop rotation or the impact of changing climate.

All four split input files must share a common prefix (e.g. "WL"), be saved in the 
source directory, and be named:
  {prefix}_plot_data.csv
    Scalar site parameters (one data row). Contains all fields from the main
    input file that do not vary over time (e.g. lat, lon, yrs_proj, species codes).

  {prefix}_mgmt_data.csv
    Management parameters. Each column is a parameter; each row is a year
    (rows 0 to yrs_proj-1, plus an optional year-0 row for thinning/mortality).
    Columns with a single value will be broadcast to all years automatically.
    An initial 'year' column (0, 1, 2, ...) is recommended but not required.

  {prefix}_tree_size_data.csv
    Tree size (age and diameter) measurements for each species. Must have at
    least 5 rows and at most yrs_proj rows. Each species contributes a pair of
    columns (e.g. age_sp1, diam_sp1).

  {prefix}_climate_cover_data.csv
    Monthly climate data and land cover fractions. proj_cover and base_cover
    are required; base_cover may be a single value (broadcast to all months).
    When NOT using the API, also include monthly Temp, Rain, and evap/pet columns
    (12 * yrs_proj rows, or a single row to broadcast). When using the API,
    only proj_cover and base_cover are needed.

See the example split files in /projects/examples/UG_TS_2016/input

_____________________STEP 2: create other required input files __________________
Other required input files are parameters for:
- biomass pools, in a file called `biomass_pool_params.csv`,
- crops, in a file called `crop_params.csv`,
- litter in a file called `litter_params.csv`,
- trees in a file called `tree_params.csv`
These should be saved in the source directory (alongside the file from STEP 1).

Default parameter files are available in `shamba/default_input`. 
hese should either:
1. be copied directly to your source directory and the files renamed to remove
    "_defaults" (e.g. `crop_params_defaults.csv` becomes `crop_params.csv`); OR
2. be used as templates to add your own data. The code expects files in the 
    formats shown. Refer to the SHAMBA methodology for definitions of the data
    points required.

Make sure the '_input.csv' file correctly attributes each tree cohort to the
relevant parameters in tree_params.csv under 'trees in baseline' and 
'trees in project'.

______________STEP 3 (optional): create project allometric functions ______________
If allometric functions not included in the SHAMBA code base are to be used, 
write these in a python file named 'project_allometry.py' in your source directory.
Note that this step requires greater python literacy than other steps.

Ensure:
1. each function returns aboveground biomass in kg C for a single tree measurement;
    using `tree_params.carbon` where necessary; AND
2. the file includes a dictionary called 'allometric' matching each allometric 
    function to a key, so that you can select it at the command line.

The allometric functions chosen at the command line will be used in the 
`get_biomass()` function in `tree_growth.py`. The functions are given a diameter at
breast height (dbh) measurement and the appropriate tree parameters for the cohort 
(provided by the user in STEP 2, above).

Functions using input data other than diameter at breast height 
(dbh) will need careful handling. A suggestion of how to handle this is included
in the example project (/projects/examples/UG_TS_2016/input/project_allometry.py)

Please note that any steps taken to use different allometry will need to be 
reproducible by a reviewer.

__________________STEP 4 (optional): create site soil & climate data _______________
Soil and climate data is either sourced from APIs, or from local csv files of your
own data. To use your own values for soil and climate data, csv files should
be added to the source directory (alongisde your input file).

The climate data csv must be called climate.csv and match the format shown in
/projects/examples/UG_TS_2016/input/climate.csv.
The soil data csv must be called soil-info.csv and match the format shown in
/projects/examples/UG_TS_2016/input/soil-info.csv.
_____________________________________________________________________________________
        """
    )

    # Generate timestamp for default project name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prompt for project name
    project_name = questionary.text(
        "Enter project name (or use auto-generated name)",
        default=f"project_{timestamp}",
    ).ask()
    arguments["project-name"] = project_name

    # Prompt for source directory
    source_directory = questionary.text(
        "Enter source directory path relative to /projects/" " (or use example)",
        default=f"examples/UG_TS_2016/input",
    ).ask()
    arguments["source-directory"] = source_directory

    split_input_data_presence = questionary.confirm(
        "Do you have split vector data saved in the source directory?", default=False
    ).ask()

    # Prompt for use-api (boolean)
    use_api = questionary.confirm("Use API for climate and soil data?", default=DEFAULT_USE_API).ask()
    arguments["use-api"] = use_api

    # Prompt for n_cohorts
    # Default to 1 if integer not provided
    n_cohorts = questionary.text("Enter number of tree cohorts (defaults to 1): ", validate=validate_integer, default="1").ask()
    arguments["n-cohorts"] = int(n_cohorts)

    # Prompt for allometric key list
    own_allometry = questionary.confirm(
        "Do you have allometric functions to use that are not in SHAMBA's default list? (if yes, please see instructions):", default=False).ask()
    own_allometric_keys = []
    allometric_keys = list(TreeGrowth.allometric.keys())
    if own_allometry == True:
        import sys
        import importlib
        
        source_dir = os.path.join(configuration.PROJECT_DIR, arguments["source-directory"])
        sys.path.insert(0, source_dir)
        project_allometry = importlib.import_module('project_allometry')
        own_allometric_keys = list(project_allometry.allometric.keys())
        
    all_allometric_keys = allometric_keys + own_allometric_keys


    # Prompt for allometric key, cohort by cohort
    cohort_allometric_keys = []

    base_selected_allometric_key = questionary.select(
        "Select an Allometric Key for the baseline species:", choices=all_allometric_keys, default=DEFAULT_ALLOMORPHY
        ).ask()
    
    cohort_allometric_keys.append(base_selected_allometric_key)

    for i in range(int(n_cohorts)):
        selected_allometric_key = questionary.select(
        "Select an Allometric Key for each species in the cohort, in the same order as the input file:", 
        choices=all_allometric_keys, default=DEFAULT_ALLOMORPHY).ask()
        cohort_allometric_keys.append(selected_allometric_key)
    arguments["allometric-keys"] = cohort_allometric_keys

    # Prompt for GWP
    gwp_keys = list(GWP_list.keys())
    selected_gwp_key = questionary.select(
        "Select Global Warming Potential values:", choices=gwp_keys, default=DEFAULT_GWP
    ).ask()
    arguments["gwp"] = GWP_list[selected_gwp_key]

    # Prompt for soil model
    soil_models = [
        Choice(title="Roth C", value=SoilModelType.ROTH_C),
        Choice(title="Example Soil Model", value=SoilModelType.EXAMPLE),
    ]

    # selected_soil_model = questionary.select(
    #     "Select a soil model:",
    #     choices=soil_models,
    #     default=SoilModelType.ROTH_C
    # ).ask()
    arguments["soil-model"] = SoilModelType.ROTH_C

    # Prompt for whether to print to stdout
    print_to_stdout = questionary.confirm("Results will be saved to csv files. Do you also want to print all to stdout?", default=False).ask()
    arguments["print-to-stdout"] = print_to_stdout

    # Prompt for input file name with default
    if split_input_data_presence:
        split_input_file_id = questionary.text(
            "Enter the prefix of the split input data files:", default="WL"
        ).ask()
        arguments["split-input-file-id"] = split_input_file_id
    else:
        input_file_name = questionary.text(
            "Enter the name of the single input file:", default="WL_input.csv"
        ).ask()
        arguments["input-file-name"] = input_file_name

    # Prompt for output title
    output_title = questionary.text(
        "Enter the title of the output file:", default="WL"
    ).ask()
    arguments["output-title"] = output_title

    # Set logging configuration
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    return arguments
