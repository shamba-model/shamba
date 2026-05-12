import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model import configuration
from model.crop_model import get_crop_bases, get_crop_projects
import model.common.constants as CONSTANTS
from model.common.data_handler import expand_single_row_data_input

#-- Expected emissions arrays -- #
# WL scenario has no fire → all emissions are zero.
WL_expected_base_emissions = [0.0] * 50
WL_expected_project_emissions = [0.0] * 50

# testB baseline also has no fire.
testB_expected_base_emissions = [0.0] * 40
# Verified in a worked example in excel.
testB_expected_project_emissions = [0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.070622,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.042233,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
]

#-- Test function -- #

@pytest.mark.parametrize("csv_input_file, expected_base_emissions, expected_project_emissions", [
    pytest.param("WL_input.csv", WL_expected_base_emissions, 
                WL_expected_project_emissions, id = "Test Case: WL"),
    pytest.param("testB_input.csv", testB_expected_base_emissions, testB_expected_project_emissions, id = "Test Case: testB"),
])

def test_crop_fire_model(csv_input_file, expected_base_emissions, expected_project_emissions):
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)
    scalar_input_data, _, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"])
    input_data = {**scalar_input_data, **mgmt_input_data}

    crop_base, _crop_par_base = get_crop_bases(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )
    crop_project, _crop_par_project = get_crop_projects(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )

    fire_base_emissions = Emit.fire_emit(
        no_of_years=N_YEARS,
        fire=mgmt_input_data["fire_on_base"],
        crop=crop_base,
        tree=[],
        litter=[],
        burn_off=mgmt_input_data["fire_off_base"],
        gwp=CONSTANTS.GWP_AR6,
    )
    fire_project_emissions = Emit.fire_emit(
        no_of_years=N_YEARS,
        fire=mgmt_input_data["fire_on_proj"],
        crop=crop_project,
        tree=[],
        litter=[],
        burn_off=mgmt_input_data["fire_off_proj"],
        gwp=CONSTANTS.GWP_AR6,
    )

    assert fire_base_emissions == pytest.approx(expected_base_emissions, rel=1e-5)
    assert fire_project_emissions == pytest.approx(expected_project_emissions, rel=1e-5)

