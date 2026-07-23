import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model.common import csv_handler
from model import configuration
import model.tree_model as TreeModel
import model.tree_params as TreeParams
import model.tree_growth as TreeGrowth
import model.common.constants as CONSTANTS
from model.common.calculate_emissions import get_int

WL_N_COHORTS = 1
WL_allometric_keys = ["chave dry", "chave dry"]
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
WL_expected_project_emissions = [5.476539,
-3.078378,
-3.253154,
-3.433162,
-3.618206,
-3.808010,
-0.638124,
-4.018925,
-4.225768,
-0.650969,
-4.419783,
-4.625534,
-4.816110,
-5.005381,
2.945191,
-4.966104,
-5.163145,
-5.321001,
-5.470332,
-5.609308,
-5.735999,
-5.848403,
-5.944467,
-6.022120,
-6.079310,
-6.114043,
-6.124429,
-6.108732,
-6.065417,
-5.993200,
-5.891097,
-5.758470,
-5.595061,
-5.401024,
85.926435,
-2.641561,
-2.832374,
-2.645545,
-2.444460,
-2.230724,
-2.006124,
-1.772587,
-1.532143,
-1.286877,
-1.038889,
-0.790245,
-0.542939,
-0.298857,
-0.059745,
0.172816,
]

TESTB_N_COHORTS = 1
testB_allometric_keys = ["chave dry","chave dry", "ryan", "markhamia"]
testB_expected_base_emissions = [0.008940113898396357, # not yet confirmed in Excel, but calculated separately and hard coded here
   -0.009102637908300728,
   -0.010442286483141057,
   -0.011822387804836385,
   -0.0010175701380445172,
   -0.013454071988146736,
   -0.015411870259866995,
   -0.01742705333420731,
   -0.019699417684040186,
   -0.022260514416071932,
   -0.025145339009472738,
   -0.028392575495939903,
   -0.032044812740706655,
   -0.036148712847146824,
   -0.040755105103445544,
   -0.04591897067998361,
   -0.05169927324985757,
   -0.05815857867923701,
   -0.06536239287976454,
   -0.07337813101664835,
   -0.08227361407739658,
   -0.09211497145158065,
   -0.10296381261135527,
   -0.11487352033874684,
   -0.12788451687657446,
   -0.14201836943782115,
   -0.15727064138365093,
   -0.17360247083707883,
   -0.19093098171777315,
   -0.2091188151102281,
   -0.22796332011994422,
   -0.24718626322092543,
   -0.266425288567455,
   -0.28522875023637484,
   -0.3030558708271895,
   -0.3192843540278421,
   -0.3332274570321082,
   -0.34416196969573964,
   -0.35136744269183867,
  -0.35417534219962393,
]
testB_expected_project_emissions = [0.000000,
0.000000,
0.000000,
-0.339945,
0.008858,
-0.032404,
-0.033640,
0.004864,
-0.039099,
-0.044191,
-0.048976,
-0.054258,
-0.056296,
-0.066511,
-0.073591,
-0.081386,
-0.089959,
-0.099377,
-0.102787,
-0.121028,
-0.133402,
-0.146902,
-0.161592,
-0.177530,
-0.182162,
-0.213317,
-0.233203,
-0.254394,
-0.276827,
0.424206,
-0.200898,
-0.280084,
-0.300536,
-0.320918,
-0.340789,
-0.359608,
-0.346230,
-0.391456,
-0.402958,
-0.410398,
]

#-- Test function -- #

@pytest.mark.parametrize("csv_input_file, N_COHORTS, allometric_keys, expected_base_emissions, expected_project_emissions", [
    pytest.param("WL_input.csv", WL_N_COHORTS, WL_allometric_keys, WL_expected_base_emissions, 
       WL_expected_project_emissions, id = "Test Case: WL"),
    pytest.param("testB_input.csv", TESTB_N_COHORTS, testB_allometric_keys, testB_expected_base_emissions, testB_expected_project_emissions, id = "Test Case: testB"),
])

def test_tree_model(csv_input_file, N_COHORTS, allometric_keys, expected_base_emissions, expected_project_emissions):
    input_csv = csv_input_file
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", input_csv)
    csv_input_data = csv_handler.get_csv_input_data(0, file_path)
    N_YEARS = int(csv_input_data["yrs_proj"])
    allometric_keys = allometric_keys

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

    tree_par_base = TreeParams.from_species_index(int(csv_input_data["species_base"]))
    tree_params_1 = TreeParams.from_species_index(int(csv_input_data["species1"]))

    thinning_base = np.zeros(N_YEARS + 1)
    thinning_base[int(csv_input_data["thin_base_yr1"])] = float(
        csv_input_data["thin_base_pc1"]
    )
    thinning_base[int(csv_input_data["thin_base_yr2"])] = float(
        csv_input_data["thin_base_pc2"]
    )

    growth_base = TreeGrowth.get_growth(
        csv_input_data,
        "species_base",
        tree_par_base,
        allometric_key=allometric_keys[0],
    )

    thinning_fraction_woody_base = np.array(
        [
            float(csv_input_data["thin_base_br"]),
            float(csv_input_data["thin_base_st"]),
        ]
    )

    mortality_base = np.array((N_YEARS + 1) * [float(csv_input_data["base_mort"])])

    mortality_fraction_woody_base = np.array(
        [
            float(csv_input_data["mort_base_br"]),
            float(csv_input_data["mort_base_st"]),
        ]
    )

    tree_base = TreeModel.from_defaults(
        tree_params=tree_params_1,
        tree_growth=growth_base,
        year_planted=0,
        stand_density=get_int(CONSTANTS.BASE_PLANT_DENSITY_KEY, csv_input_data),
        thinning=thinning_base,
        thinning_fraction_woody=thinning_fraction_woody_base,
        mortality=mortality_base,
        mortality_fraction_woody=mortality_fraction_woody_base,
        no_of_years=N_YEARS,
    )

    tree_params = TreeParams.create_tree_params_from_species_index(
        csv_input_data, N_COHORTS
    )
    tree_growths = TreeGrowth.create_tree_growths(
        csv_input_data, tree_params, allometric_keys, N_COHORTS
    )
    
    thinning_project = np.zeros(N_YEARS + 1)
    thinning_project[int(csv_input_data["thin_proj_yr1"])] = float(
        csv_input_data["thin_proj_pc1"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr2"])] = float(
        csv_input_data["thin_proj_pc2"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr3"])] = float(
        csv_input_data["thin_proj_pc3"]
    )
    thinning_project[int(csv_input_data["thin_proj_yr4"])] = float(
        csv_input_data["thin_proj_pc4"]
    )
    thinning_fraction_woody_project = np.array(
        [
            float(csv_input_data["thin_proj_br"]),
            float(csv_input_data["thin_proj_st"]),
        ]
    )
    mortality_project = np.array((N_YEARS + 1) * [float(csv_input_data["proj_mort"])])
    mortality_fraction_woody_project = np.array(
        [
            float(csv_input_data["mort_proj_br"]),
            float(csv_input_data["mort_proj_st"]),
        ]
    )

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=csv_input_data,
        tree_params=tree_params,
        growths=tree_growths,
        thinning_project=thinning_project,
        thinning_fraction_woody_project=thinning_fraction_woody_project,
        mortality_project=mortality_project,
        mortality_fraction_woody_project=mortality_fraction_woody_project,
        no_of_years=N_YEARS,
        cohort_count=N_COHORTS,
    )


    tree_base_emissions = Emit.create(
        no_of_years=N_YEARS, tree=[tree_base], fire=fire_base, burn_off=burn_off_base,
    )
    tree_project_emissions = Emit.create(
        no_of_years=N_YEARS, tree=tree_projects, fire=fire_project, burn_off=burn_off_project,
    )

    assert tree_base_emissions == pytest.approx(expected_base_emissions, rel=1e-3 )
    assert tree_project_emissions == pytest.approx(expected_project_emissions, rel=1e-3)
    # These tests require lower accuracy than other tests. This is because the results are dependent on the parameters of the fitted equations, 
    # which may vary slightly between the code calculations and the expected results, which were calculated separately and hard coded here.
