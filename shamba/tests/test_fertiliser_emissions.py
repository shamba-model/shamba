import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.litter as LitterModel

#-- Expected emissions arrays -- #
WL_expected_base_emissions = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
WL_expected_project_emissions = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

testB_expected_base_emissions = [0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
]
testB_expected_project_emissions = [0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
0.0000,
0.0000,
0.7636,
]

#-- Test function -- #

@pytest.mark.parametrize("csv_input_file, expected_base_emissions, expected_project_emissions", [
    pytest.param("WL_input.csv", WL_expected_base_emissions, 
                WL_expected_project_emissions, id = "Test Case: WL"),
    pytest.param("testB_input.csv", testB_expected_base_emissions, testB_expected_project_emissions, id = "Test Case: testB"),
])

def test_fertiliser_model(csv_input_file, expected_base_emissions, expected_project_emissions):
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

    base_sf_qty_vec = np.zeros(N_YEARS)
    base_sf_n_vec = np.zeros(N_YEARS)
    base_sf_int = int(csv_input_data["base_sf_int"])
    if base_sf_int > 0:
        base_sf_qty_vec[::base_sf_int] = float(csv_input_data["base_sf_qty"])
        base_sf_n_vec[::base_sf_int] = float(csv_input_data["base_sf_n"])

    proj_sf_qty_vec = np.zeros(N_YEARS)
    proj_sf_n_vec = np.zeros(N_YEARS)
    proj_sf_int = int(csv_input_data["proj_sf_int"])
    if proj_sf_int > 0:
        proj_sf_qty_vec[::proj_sf_int] = float(csv_input_data["proj_sf_qty"])
        proj_sf_n_vec[::proj_sf_int] = float(csv_input_data["proj_sf_n"])

    synthetic_fertiliser_base = LitterModel.synthetic_fertiliser(
        quantity_vector=base_sf_qty_vec,
        nitrogen_vector=base_sf_n_vec,
    )
    synthetic_fertiliser_project = LitterModel.synthetic_fertiliser(
        quantity_vector=proj_sf_qty_vec,
        nitrogen_vector=proj_sf_n_vec,
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

    fertiliser_base_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_base], fire=fire_base, burn_off=burn_off_base
    )
    fertiliser_project_emissions = Emit.create(
        no_of_years=N_YEARS, fert=[synthetic_fertiliser_project], fire=fire_project, burn_off=burn_off_project
    )

    assert fertiliser_base_emissions == pytest.approx(
        expected_base_emissions, rel=1e-4)

    assert fertiliser_project_emissions == pytest.approx(
        expected_project_emissions, rel=1e-4)

