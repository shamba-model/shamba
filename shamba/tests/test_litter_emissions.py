import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.litter as LitterModel

#-- Expected emissions arrays -- #
WL_expected_base_emissions = [0.0000,
0.0000,
0.0000,
0.0000,
0.0000,
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
WL_expected_project_emissions = [0.0000,
0.0000,
0.0000,
0.0000,
0.0000,
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

testB_expected_base_emissions = [1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
]
testB_expected_project_emissions = [2.57349,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
0.00000,
0.00000,
0.00000,
0.00000,
2.57349,
0.00000,
0.00000,
0.00000,
0.00000,
1.38224,
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

def test_litter_model(csv_input_file, expected_base_emissions, expected_project_emissions):
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])

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

    base_lit_vec = np.zeros(N_YEARS)
    base_lit_int = int(csv_input_data["base_lit_int"])
    if base_lit_int > 0:
        base_lit_vec[::base_lit_int] = float(csv_input_data["base_lit_qty"])

    proj_lit_vec = np.zeros(N_YEARS)
    proj_lit_int = int(csv_input_data["proj_lit_int"])
    if proj_lit_int > 0:
        proj_lit_vec[::proj_lit_int] = float(csv_input_data["proj_lit_qty"])

    litter_external_base = LitterModel.from_defaults(litter_vector=base_lit_vec)
    litter_external_project = LitterModel.from_defaults(litter_vector=proj_lit_vec)

    litter_base_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_base], fire=fire_base, burn_off=burn_off_base,
    )
    litter_project_emissions = Emit.create(
        no_of_years=N_YEARS, litter=[litter_external_project], fire=fire_project, burn_off=burn_off_project
    )

    assert litter_base_emissions == pytest.approx(expected_base_emissions, rel=1e-5)
    assert litter_project_emissions == pytest.approx(expected_project_emissions, rel=1e-5)
