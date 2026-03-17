import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model import configuration
import model.tree_model as TreeModel
import model.tree_params as TreeParams
import model.tree_growth as TreeGrowth
from model.common.data_handler import expand_single_row_data_input

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
   -0.005289552029268359,
   -0.006570496126139579,
   -0.007452309050098124,
   0.003916609716804662,
   -0.007881478073596893,
   -0.00974645937868921,
   -0.01102635213634245,
   -0.012467285467020205,
   -0.014088526559133632,
   -0.015911276393376098,
   -0.017958747819204996,
   -0.020256203537976913,
   -0.022830935187560863,
   -0.025712159028650584,
   -0.028930796737141613,
   -0.03251910136366279,
   -0.03651007855264328,
   -0.04093664167291318,
   -0.04583042687995835,
   -0.05122018094829765,
   -0.05712962216873538,
   -0.0635746646723725,
   -0.07055989226909143,
   -0.07807417371374997,
   -0.08608533336274822,
   -0.09453383742029675,
   -0.1033255360374055,
   -0.11232362597972942,
   -0.12134017719080091,
   -0.13012780521905543,
   -0.1383723673003325,
   -0.14568789473105662,
   -0.15161530691171513,
   -0.1556267122317543,
   -0.15713718491468504,
   -0.15552568754730373,
   -0.15016615715258833,
   -0.14046859974582754,
  -0.12592835465828683,
]
testB_expected_project_emissions = [0.000000,
0.000000,
0.000000,
-0.339945,
0.008858,
-0.017152,
-0.029258,
0.023502,
-0.018485,
-0.023666,
-0.026264,
-0.029122,
-0.049062,
-0.035715,
-0.039503,
-0.043654,
-0.048198,
-0.053162,
-0.089492,
-0.064458,
-0.070836,
-0.077725,
-0.085133,
-0.093058,
-0.157910,
-0.110373,
-0.119666,
-0.129269,
-0.139052,
0.575760,
-0.157603,
-0.133902,
-0.140363,
-0.145708,
-0.149490,
-0.151184,
-0.287329,
-0.145870,
-0.137508,
-0.124406,
]

#-- Test function -- #

@pytest.mark.parametrize("csv_input_file, N_COHORTS, allometric_keys, expected_base_emissions, expected_project_emissions", [
    pytest.param("WL_input.csv", WL_N_COHORTS, WL_allometric_keys, WL_expected_base_emissions, 
       WL_expected_project_emissions, id = "Test Case: WL"),
    pytest.param("testB_input.csv", TESTB_N_COHORTS, testB_allometric_keys, testB_expected_base_emissions, testB_expected_project_emissions, id = "Test Case: testB"),
])

def test_tree_model(csv_input_file, N_COHORTS, allometric_keys, expected_base_emissions, expected_project_emissions):
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)

    scalar_input_data, tree_size_data, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"][0])

    fire_base = mgmt_input_data["fire_on_base"]
    fire_project = mgmt_input_data["fire_on_proj"]
    burn_off_base = bool(np.any(mgmt_input_data["fire_off_base"]))
    burn_off_project = bool(np.any(mgmt_input_data["fire_off_proj"]))

    tree_par_base = TreeParams.from_species_index(int(scalar_input_data["base_species1"][0]))
    tree_params_1 = TreeParams.from_species_index(int(scalar_input_data["proj_species1"][0]))

    # growth_input merges scalars, tree size, and an alias so create_tree_params_from_species_index
    # can find "species1" (its expected key for the first project cohort)
    growth_input = {
        **scalar_input_data,
        **tree_size_data,
        "species1": scalar_input_data["proj_species1"],
    }

    growth_base = TreeGrowth.get_growth(
        growth_input, "base_species1", tree_par_base, allometric_key=allometric_keys[0]
    )

    thinning_base = mgmt_input_data["thin_base_cohort1"]
    thinning_fraction_left_base = np.array([
        1,
        mgmt_input_data["thin_base_br_cohort1"][0],
        mgmt_input_data["thin_base_st_cohort1"][0],
        1, 1,
    ])
    mortality_base = mgmt_input_data["mort_base_cohort1"]
    mortality_fraction_left_base = np.array([
        1,
        mgmt_input_data["mort_base_br_cohort1"][0],
        mgmt_input_data["mort_base_st_cohort1"][0],
        1, 1,
    ])

    tree_base = TreeModel.from_defaults(
        tree_params=tree_params_1,
        tree_growth=growth_base,
        year_planted=0,
        stand_density=int(scalar_input_data["base_plant_dens1"][0]),
        thinning=thinning_base,
        thinning_fraction=thinning_fraction_left_base,
        mortality=mortality_base,
        mortality_fraction=mortality_fraction_left_base,
        no_of_years=N_YEARS,
    )

    tree_params = TreeParams.create_tree_params_from_species_index(growth_input, N_COHORTS)
    tree_growths = TreeGrowth.create_tree_growths(growth_input, tree_params, allometric_keys, N_COHORTS)

    thinning_project = mgmt_input_data["thin_proj_cohort1"]
    thinning_fraction_left_project = np.array([
        1,
        mgmt_input_data["thin_proj_br_cohort1"][0],
        mgmt_input_data["thin_proj_st_cohort1"][0],
        1, 1,
    ])
    mortality_project = mgmt_input_data["mort_proj_cohort1"]
    mortality_fraction_left_project = np.array([
        1,
        mgmt_input_data["mort_proj_br_cohort1"][0],
        mgmt_input_data["mort_proj_st_cohort1"][0],
        1, 1,
    ])

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=growth_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinning_project=thinning_project,
        thinning_fraction_left_project=thinning_fraction_left_project,
        mortality_project=mortality_project,
        mortality_fraction_left_project=mortality_fraction_left_project,
        no_of_years=N_YEARS,
        cohort_count=N_COHORTS,
    )

    tree_base_emissions = Emit.create(
        no_of_years=N_YEARS, tree=[tree_base], fire=fire_base, burn_off=burn_off_base,
    )
    tree_project_emissions = Emit.create(
        no_of_years=N_YEARS, tree=tree_projects, fire=fire_project, burn_off=burn_off_project,
    )

    assert tree_base_emissions == pytest.approx(expected_base_emissions, rel=1e-3)
    assert tree_project_emissions == pytest.approx(expected_project_emissions, rel=1e-3)
    # These tests require lower accuracy than other tests. This is because the results are dependent on the parameters of the fitted equations,
    # which may vary slightly between the code calculations and the expected results, which were calculated separately and hard coded here.
