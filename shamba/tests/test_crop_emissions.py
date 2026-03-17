import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
from model.crop_model import get_crop_bases, get_crop_projects
from model.common.data_handler import expand_single_row_data_input

#-- Expected emissions arrays -- #
WL_expected_base_emissions = [0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
        ]
WL_expected_project_emissions = [0.22413,
0.22413,
0.22413,
0.22413,
0.22413,
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

testB_expected_base_emissions = [0.0000,
0.24072,
0.24072,
0.24072,
0.24072,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
0.10581,
]
testB_expected_project_emissions = [0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.21471,
0.19677,
0.19677,
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
0.11064,
0.11064,
0.11064,
0.11064,
0.12334,
0.11064,
0.11064,
0.11064,
0.11064,
0.11064,
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
def test_crop_model(csv_input_file, expected_base_emissions, expected_project_emissions): 
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    scalar_input_data, _, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    input_data = {**scalar_input_data, **mgmt_input_data}

    crop_base, _crop_par_base = get_crop_bases(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )
    crop_project, _crop_par_project = get_crop_projects(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=3
    )

    base_fire_interval = int(csv_input_data["fire_int_base"])
    if base_fire_interval == 0:
        fire_base = np.zeros(N_YEARS)
    else:
        fire_base = np.zeros(N_YEARS)
        fire_base[::base_fire_interval] = int(csv_input_data["fire_pres_base"])

    proj_fire_interval = int(csv_input_data["fire_int_proj"])
    if proj_fire_interval == 0:
        fire_project = np.zeros(N_YEARS)
    else:
        fire_project = np.zeros(N_YEARS)
        fire_project[::proj_fire_interval] = int(csv_input_data["fire_pres_proj"])

    base_fire_off_field = int(csv_input_data["fire_off_base"])
    if base_fire_off_field == 1:
        burn_off_base = True
    else:        burn_off_base = False

    proj_fire_off_field = int(csv_input_data["fire_off_proj"])
    if proj_fire_off_field == 1:
        burn_off_project = True
    else:        burn_off_project = False

    crop_base_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_base, fire=fire_base, burn_off=burn_off_base)
    crop_project_emissions = Emit.create(
        no_of_years=N_YEARS, crop=crop_project, fire=fire_project, burn_off=burn_off_project
    )
    assert crop_base_emissions == pytest.approx(
        expected_base_emissions, rel=1e-4
    )
    assert crop_project_emissions == pytest.approx(
        expected_project_emissions, rel=1e-4
    )
