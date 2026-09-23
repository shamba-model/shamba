import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model import configuration
import model.tree_model as TreeModel
import model.tree_params as TreeParams
import model.tree_growth as TreeGrowth
from model.common.data_handler import expand_single_row_data_input, validate_all_grouped_headers
from model.monte_carlo.distribution_handler import DistributionSpec
from model.monte_carlo.sampler import sample_species_params

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
testB_allometric_keys = ["chave dry", "chave dry"]
# TODO: confirm testB_expected_base_emissions against the SHAMBA Excel reference model.
# Values were calculated separately from the code but have not yet been verified in Excel.
# This is the highest-priority independent check needed for the tree sub-model.
testB_expected_base_emissions = [0.008940113898396357,
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
# TODO: confirm testB_expected_project_emissions against the SHAMBA Excel reference model.
# Same status as base emissions above — calculated separately, not yet Excel-verified.
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
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)

    scalar_input_data, tree_size_data, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"].item())

    fire_base = mgmt_input_data["fire_on_base"]
    fire_project = mgmt_input_data["fire_on_proj"]
    burn_off_base = bool(np.any(mgmt_input_data["fire_off_base"]))
    burn_off_project = bool(np.any(mgmt_input_data["fire_off_proj"]))

    tree_species_data = TreeParams.load_tree_species_data()
    pool_species_data = TreeModel.load_biomass_pool_species_data()

    tree_par_base = TreeParams.from_species_index(
        int(scalar_input_data["base_species1"][0]), species_data=tree_species_data
    )
    tree_params_1 = TreeParams.from_species_index(
        int(scalar_input_data["proj_species1"][0]), species_data=tree_species_data
    )

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
        tree_params=tree_par_base,
        tree_growth=growth_base,
        year_planted=0,
        stand_density=int(scalar_input_data["base_plant_dens1"][0]),
        thinning=thinning_base,
        thinning_fraction=thinning_fraction_left_base,
        mortality=mortality_base,
        mortality_fraction=mortality_fraction_left_base,
        no_of_years=N_YEARS,
        pool_species_data=pool_species_data,
    )

    tree_params = TreeParams.create_tree_params_from_species_index(
        growth_input, N_COHORTS, species_data=tree_species_data
    )
    tree_growths = TreeGrowth.create_tree_growths(growth_input, tree_params, allometric_keys, N_COHORTS)

    thinnings_project = [mgmt_input_data["thin_proj_cohort1"]]
    thinning_fractions_project = [np.array([
        1,
        mgmt_input_data["thin_proj_br_cohort1"][0],
        mgmt_input_data["thin_proj_st_cohort1"][0],
        1, 1,
    ])]
    mortalities_project = [mgmt_input_data["mort_proj_cohort1"]]
    mortality_fractions_project = [np.array([
        1,
        mgmt_input_data["mort_proj_br_cohort1"][0],
        mgmt_input_data["mort_proj_st_cohort1"][0],
        1, 1,
    ])]

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=growth_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinnings_project=thinnings_project,
        thinning_fractions_project=thinning_fractions_project,
        mortalities_project=mortalities_project,
        mortality_fractions_project=mortality_fractions_project,
        no_of_years=N_YEARS,
        cohort_count=N_COHORTS,
        type="proj",
        pool_species_data=pool_species_data,
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


def test_create_tree_projects_uses_per_cohort_thinning():
    """Each project cohort must receive its own thinning array, not a shared one."""
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    scalar_input_data, tree_size_data, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"].item())

    # Build a two-cohort scenario reusing WL species for both cohorts.
    # create_tree_params_from_species_index reads "species{N}";
    # create_tree_growths reads "proj_species{N}" at allometric_keys[N] (index 0 = base).
    growth_input = {
        **scalar_input_data,
        **tree_size_data,
        "species1": scalar_input_data["proj_species1"],
        "species2": scalar_input_data["proj_species1"],
        "proj_species2": scalar_input_data["proj_species1"],
        "proj_plant_yr2": scalar_input_data["proj_plant_yr1"],
        "proj_plant_dens2": scalar_input_data["proj_plant_dens1"],
    }
    tree_species_data = TreeParams.load_tree_species_data()
    pool_species_data = TreeModel.load_biomass_pool_species_data()
    tree_params = TreeParams.create_tree_params_from_species_index(
        growth_input, 2, species_data=tree_species_data
    )
    tree_growths = TreeGrowth.create_tree_growths(
        growth_input, tree_params, ["chave dry", "chave dry", "chave dry"], 2
    )

    thinning_cohort1 = mgmt_input_data["thin_proj_cohort1"].copy()
    thinning_cohort2 = mgmt_input_data["thin_proj_cohort1"].copy()
    # Give cohort 2 a different thinning value at year 0 so the arrays are distinguishable.
    thinning_cohort2[0] = 0.99

    fraction = np.array([1, 0.0, 0.0, 1, 1])

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=growth_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinnings_project=[thinning_cohort1, thinning_cohort2],
        thinning_fractions_project=[fraction, fraction],
        mortalities_project=[
            mgmt_input_data["mort_proj_cohort1"],
            mgmt_input_data["mort_proj_cohort1"],
        ],
        mortality_fractions_project=[fraction, fraction],
        no_of_years=N_YEARS,
        cohort_count=2,
        type="proj",
        pool_species_data=pool_species_data,
    )

    assert tree_projects[0].thinning[0] != tree_projects[1].thinning[0], (
        "Cohort 1 and cohort 2 should have different thinning arrays"
    )
    np.testing.assert_array_equal(tree_projects[0].thinning, thinning_cohort1)
    np.testing.assert_array_equal(tree_projects[1].thinning, thinning_cohort2)

def test_create_tree_baselines_uses_per_cohort_thinning():
    """Each baseline cohort must receive its own thinning array, not a shared one."""
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    scalar_input_data, tree_size_data, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"].item())

    # Build a two-cohort scenario reusing WL species for both cohorts.
    # create_tree_params_from_species_index reads "species{N}";
    # create_tree_growths reads "proj_species{N}" at allometric_keys[N] (index 0 = base).
    growth_input = {
        **scalar_input_data,
        **tree_size_data,
        "species1": scalar_input_data["base_species1"],
        "species2": scalar_input_data["base_species1"],
        "proj_species2": scalar_input_data["base_species1"],
        "proj_plant_yr2": scalar_input_data["base_plant_yr1"],
        "proj_plant_dens2": scalar_input_data["base_plant_dens1"],
    }
    tree_species_data = TreeParams.load_tree_species_data()
    pool_species_data = TreeModel.load_biomass_pool_species_data()
    tree_params = TreeParams.create_tree_params_from_species_index(
        growth_input, 2, species_data=tree_species_data
    )
    tree_growths = TreeGrowth.create_tree_growths(
        growth_input, tree_params, ["chave dry", "chave dry", "chave dry"], 2
    )

    thinning_cohort1 = mgmt_input_data["thin_proj_cohort1"].copy()
    thinning_cohort2 = mgmt_input_data["thin_proj_cohort1"].copy()
    # Give cohort 2 a different thinning value at year 0 so the arrays are distinguishable.
    thinning_cohort2[0] = 0.99

    fraction = np.array([1, 0.0, 0.0, 1, 1])

    tree_baselines = TreeModel.create_tree_projects(
        csv_input_data=growth_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinnings_project=[thinning_cohort1, thinning_cohort2],
        thinning_fractions_project=[fraction, fraction],
        mortalities_project=[
            mgmt_input_data["mort_proj_cohort1"],
            mgmt_input_data["mort_proj_cohort1"],
        ],
        mortality_fractions_project=[fraction, fraction],
        no_of_years=N_YEARS,
        cohort_count=2,
        type="proj",
        pool_species_data=pool_species_data,
    )

    assert tree_baselines[0].thinning[0] != tree_baselines[1].thinning[0], (
        "Cohort 1 and cohort 2 should have different thinning arrays"
    )
    np.testing.assert_array_equal(tree_baselines[0].thinning, thinning_cohort1)
    np.testing.assert_array_equal(tree_baselines[1].thinning, thinning_cohort2)


def test_validation_requires_per_cohort_thinning_when_second_cohort_present():
    """validate_all_grouped_headers must flag missing thinning/mortality for cohort 2."""
    data = {
        "proj_species1": np.array([1]),
        "proj_species2": np.array([2]),
        # cohort 1 thinning present
        "thin_proj_cohort1": np.zeros(5),
        "thin_proj_br_cohort1": np.array([0.0]),
        "thin_proj_st_cohort1": np.array([0.0]),
        "mort_proj_cohort1": np.zeros(5),
        "mort_proj_br_cohort1": np.array([0.0]),
        "mort_proj_st_cohort1": np.array([0.0]),
        # cohort 2 thinning deliberately absent
    }
    errors = validate_all_grouped_headers(data)
    missing_keys = [e for e in errors if "cohort2" in e]
    assert len(missing_keys) == 6, (
        f"Expected 6 missing-cohort-2 errors (one per thinning/mortality key), got: {missing_keys}"
    )


def _write_pool_params_csv(tmp_path, branch_al, stem_al):
    """Minimal single-species biomass_pool_params.csv, branch/stem AL as given."""
    path = tmp_path / "biomass_pool_params.csv"
    path.write_text(
        "Sc,pool,TO,AL,THf,DTf\n"
        "1,leaf,1,0.1,1,1\n"
        f"1,branch,0.05,{branch_al},0,0\n"
        f"1,stem,0,{stem_al},0,0\n"
        "1,croot,0,0,1,1\n"
        "1,froot,0.8,0.1,1,1\n"
    )
    return str(path)


def _wl_tree_par_and_growth():
    """Species-1 TreeParams and TreeGrowth from the WL fixture, for pool-alloc tests
    that don't care about the rest of the intervention input."""
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", "WL_input.csv")
    scalar_input_data, tree_size_data, _, _ = expand_single_row_data_input(file_path)
    tree_species_data = TreeParams.load_tree_species_data()
    tree_par = TreeParams.from_species_index(
        int(scalar_input_data["base_species1"][0]), species_data=tree_species_data
    )
    growth_input = {
        **scalar_input_data,
        **tree_size_data,
        "species1": scalar_input_data["proj_species1"],
    }
    growth = TreeGrowth.get_growth(
        growth_input, "base_species1", tree_par, allometric_key="chave dry"
    )
    no_of_years = int(scalar_input_data["yrs_proj"].item())
    stand_density = int(scalar_input_data["base_plant_dens1"][0])
    return tree_par, growth, no_of_years, stand_density


def test_load_biomass_pool_species_data_rejects_branch_stem_mismatch(tmp_path):
    """Branch + stem 'AL' must sum to 1 (SHAMBA_ModelDescription_v1.2, Sec 4.3.2:
    alstem/albranch are a split of total above-ground biomass)."""
    bad_path = _write_pool_params_csv(tmp_path, branch_al=0.5, stem_al=0.6)
    with pytest.raises(ValueError, match=r"branch and stem 'AL'"):
        TreeModel.load_biomass_pool_species_data(filename=bad_path)


def test_load_biomass_pool_species_data_accepts_branch_stem_summing_to_one(tmp_path):
    good_path = _write_pool_params_csv(tmp_path, branch_al=0.31, stem_al=0.69)
    species_data = TreeModel.load_biomass_pool_species_data(filename=good_path)
    assert species_data[1]["alloc"][1] == pytest.approx(0.31)
    assert species_data[1]["alloc"][2] == pytest.approx(0.69)


def test_from_defaults_derives_branch_alloc_from_stem():
    """from_defaults() must derive branch = 1 - stem itself, not just trust a
    validated species-lookup value — this is what keeps the identity true once
    MC sampling can perturb stem independently of whatever branch happened to be."""
    tree_par, growth, no_of_years, stand_density = _wl_tree_par_and_growth()
    pool_species_data = TreeModel.load_biomass_pool_species_data()

    # Deliberately inconsistent branch value (would fail load-time validation
    # if it came from a real CSV) — proves from_defaults() ignores it and
    # derives branch from stem regardless of pool_species_data.
    tampered = {k: dict(v) for k, v in pool_species_data.items()}
    tampered[tree_par.species]["alloc"] = np.array(
        [0.1, 0.99, 0.69, 0.0, 0.1]
    )  # leaf, branch(fake), stem, croot, froot

    tree = TreeModel.from_defaults(
        tree_params=tree_par,
        tree_growth=growth,
        no_of_years=no_of_years,
        stand_density=stand_density,
        pool_species_data=tampered,
    )
    assert tree.alloc[1] == pytest.approx(1 - tree.alloc[2])
    assert tree.alloc[1] == pytest.approx(0.31)  # 1 - 0.69, not the fake 0.99


def test_from_defaults_derives_branch_alloc_after_species_param_sampling():
    """pool_alloc_sp{N} sampling perturbs stem independently of branch; the
    branch+stem=1 identity must still hold on every sampled draw."""
    tree_par, growth, no_of_years, stand_density = _wl_tree_par_and_growth()
    pool_species_data = TreeModel.load_biomass_pool_species_data()

    rng = np.random.default_rng(42)
    spec = DistributionSpec(
        parameter=f"pool_alloc_sp{tree_par.species}",
        distribution="uniform",
        spread_lower=0.2,
        spread_upper=0.2,
        min_abs=None,
    )
    samples = sample_species_params(
        base_params=pool_species_data,
        distributions={f"pool_alloc_sp{tree_par.species}": spec},
        param_fields=TreeModel.BIOMASS_POOL_PARAM_FIELDS,
        key_prefix="pool",
        n_samples=10,
        rng=rng,
    )

    base_stem = pool_species_data[tree_par.species]["alloc"][2]
    saw_perturbed_stem = False
    for sample in samples:
        tree = TreeModel.from_defaults(
            tree_params=tree_par,
            tree_growth=growth,
            no_of_years=no_of_years,
            stand_density=stand_density,
            pool_species_data=sample,
        )
        assert tree.alloc[1] == pytest.approx(1 - tree.alloc[2])
        assert tree.alloc[1] + tree.alloc[2] == pytest.approx(1.0)
        if tree.alloc[2] != pytest.approx(base_stem):
            saw_perturbed_stem = True

    assert saw_perturbed_stem, (
        "Expected pool_alloc_sp sampling to actually perturb stem across samples"
    )
