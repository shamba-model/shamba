from typing import Dict, Union, List, NamedTuple, Tuple, Any, Callable
from toolz import get, compose  # type: ignore
from copy import deepcopy
import numpy as np

import model.climate as Climate
import model.crop_model as CropModel
import model.litter as LitterModel
import model.soil_params as SoilParams
import model.tree_growth as TreeGrowth
import model.tree_model as TreeModel
import model.tree_params as TreeParams
import model.crop_params as CropParams
import model.emit as Emit
import model.common.constants as CONSTANTS

import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
from model.soil_models.soil_model_types import (
    SoilModelType,
    ForwardSoilModelData,
    InverseSoilModelData,
)

get_float: Callable[[str, Dict[str, Any]], float] = compose(float, get)  # type: ignore
get_int: Callable[[str, Dict[str, Any]], int] = compose(int, get)  # type: ignore


def get_location(year_input: Dict[str, Any]) -> Tuple[float, float]:
    return (
        get_float(CONSTANTS.LOCATION_LATITIUDE_KEY, year_input),
        get_float(CONSTANTS.LOCATION_LONGITUDE_KEY, year_input),
    )


def populate_thinning_array(
    intervention_input: Dict[str, Union[float, int]],
    key_pairs: List[Tuple[str, str]],
    no_of_years: int,
):
    thinning_array = np.zeros(no_of_years + 1)
    for year_key, percent_key in key_pairs:
        year = get_int(year_key, intervention_input)
        percent = get_float(percent_key, intervention_input)
        thinning_array[year] = percent
    return thinning_array


# ----------
# TREE MODEL
# ----------
class GetTreeModelReturnData(NamedTuple):
    tree_base: TreeModel.TreeModel
    tree_projects: List[TreeModel.TreeModel]
    tree_growths: List[TreeGrowth.TreeGrowth]


def get_tree_model_data(
    intervention_input: Dict[str, Union[float, int]],
    no_of_years: int,
    no_of_cohorts: int,
    allometry: List[str],
) -> GetTreeModelReturnData:
    # Linking tree cohort parameteres
    tree_par_base = TreeParams.from_species_index(
        get_int(CONSTANTS.SPECIES_BASE_KEY, intervention_input)
    )
    tree_params_1 = TreeParams.from_species_index(
        get_int(CONSTANTS.SPECIES_1_KEY, intervention_input)
    )

    tree_params = TreeParams.create_tree_params_from_species_index(
        intervention_input, no_of_cohorts
    )

    # Linking tree growth
    # Baseline growth: single species # identified in species_base
    growth_base = TreeGrowth.get_growth(
        intervention_input,
        CONSTANTS.SPECIES_BASE_KEY,
        tree_par_base,
        allometric_key=allometry[0]
    )

    # Intervention growth: species # identified in no_of_cohorts different cases,
    # so uses a wrapper function to identify species info for each cohort
    tree_growths = TreeGrowth.create_tree_growths(
        intervention_input, tree_params, allometry, no_of_cohorts
    )

    # Specify thinning regime and fraction left in field (lif)
    # baseline thinning regime
    # (add line of thinning[yr] = % thinned for each event)
    thinning_base = populate_thinning_array(
        intervention_input,
        [
            (CONSTANTS.THINNING_BASE_YEAR_1_KEY, CONSTANTS.THINNING_BASE_PC1_KEY),
            (CONSTANTS.THINNING_BASE_YEAR_2_KEY, CONSTANTS.THINNING_BASE_PC2_KEY),
        ],
        no_of_years,
    )

    thinning_project = populate_thinning_array(
        intervention_input,
        [
            (CONSTANTS.THINNING_PROJECT_YEAR_1_KEY, CONSTANTS.THINNING_PROJECT_PC1_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_2_KEY, CONSTANTS.THINNING_PROJECT_PC2_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_3_KEY, CONSTANTS.THINNING_PROJECT_PC3_KEY),
            (CONSTANTS.THINNING_PROJECT_YEAR_4_KEY, CONSTANTS.THINNING_PROJECT_PC4_KEY),
        ],
        no_of_years,
    )

    # Baseline fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_base = np.array(
        [
            1,
            get_float(CONSTANTS.THINNING_BASE_BR_KEY, intervention_input),
            get_float(CONSTANTS.THINNING_BASE_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Project fraction of thinning left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    thinning_fraction_left_project = np.array(
        [
            1,
            get_float(CONSTANTS.THINNING_PROJECT_BR_KEY, intervention_input),
            get_float(CONSTANTS.THINNING_PROJECT_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Specify mortality regime and fraction left in field (lif)
    # Baseline yearly mortality
    mortality_base = np.array(
        (no_of_years + 1)
        * [get_float(CONSTANTS.BASE_MORTALITY_KEY, intervention_input)]
    )

    # Project yearly mortality
    mortality_project = np.array(
        (no_of_years + 1)
        * [get_float(CONSTANTS.PROJECT_MORTALITY_KEY, intervention_input)]
    )

    # Baseline fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_base = np.array(
        [
            1,
            get_float(CONSTANTS.MORTALITY_BASE_BR_KEY, intervention_input),
            get_float(CONSTANTS.MORTALITY_BASE_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # Project fraction of dead biomass left in the field
    # specify vector = array[(leaf,branch,stem,course root,fine root)].
    # 1 = 100% left in field. Leaf and roots assumed 100%.
    # (can specify for individual years) using above code for thinning_project.
    mortality_fraction_left_project = np.array(
        [
            1,
            get_float(CONSTANTS.MORTALITY_PROJECT_BR_KEY, intervention_input),
            get_float(CONSTANTS.MORTALITY_PROJECT_ST_KEY, intervention_input),
            1,
            1,
        ]
    )

    # RUN TREE MODEL
    # Trees planted in baseline (stand_dens must be at least 1)
    tree_base = TreeModel.from_defaults(
        tree_params=tree_params_1,
        tree_growth=growth_base,
        year_planted=get_int(CONSTANTS.BASE_PLANT_YEAR_KEY, intervention_input),
        stand_density=get_int(CONSTANTS.BASE_PLANT_DENSITY_KEY, intervention_input),
        thinning=thinning_base,
        thinning_fraction=thinning_fraction_left_base,
        mortality=mortality_base,
        mortality_fraction=mortality_fraction_left_base,
        no_of_years=no_of_years,
    )

    tree_projects = TreeModel.create_tree_projects(
        csv_input_data=intervention_input,
        tree_params=tree_params,
        growths=tree_growths,
        thinning_project=thinning_project,
        thinning_fraction_left_project=thinning_fraction_left_project,
        mortality_project=mortality_project,
        mortality_fraction_left_project=mortality_fraction_left_project,
        no_of_years=no_of_years,
        cohort_count=no_of_cohorts,
    )

    return GetTreeModelReturnData(
        tree_base=tree_base, tree_projects=tree_projects, tree_growths=tree_growths
    )


# ----------
# FIRE MODEL
# ----------
class GetFireModelReturnData(NamedTuple):
    fire_base: np.ndarray
    fire_project: np.ndarray
    fire_off_base: np.ndarray
    fire_off_proj: np.ndarray


def get_fire_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetFireModelReturnData:
    return GetFireModelReturnData(
        fire_base=np.array(intervention_input["fire_on_base"]),
        fire_project=np.array(intervention_input["fire_on_proj"]),
        fire_off_base=np.array(intervention_input["fire_off_base"]),
        fire_off_proj=np.array(intervention_input["fire_off_proj"]),
    )


class GetLitterModelReturnData(NamedTuple):
    litter_external_base: LitterModel.LitterModelData
    litter_external_project: LitterModel.LitterModelData
    synthetic_fertiliser_base: LitterModel.LitterModelData
    synthetic_fertiliser_project: LitterModel.LitterModelData


def get_litter_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetLitterModelReturnData:
    
    litter_external_base = LitterModel.from_defaults(
        litter_vector=intervention_input["base_lit_qty"],
    )
    litter_external_project = LitterModel.from_defaults(
        litter_vector=intervention_input["proj_lit_qty"],
    )
    synthetic_fertiliser_base = LitterModel.synthetic_fertiliser(
        quantity_vector=intervention_input["base_sf_qty"],
        nitrogen_vector=intervention_input["base_sf_n"],
    )
    synthetic_fertiliser_project = LitterModel.synthetic_fertiliser(
        quantity_vector=intervention_input["proj_sf_qty"],
        nitrogen_vector=intervention_input["proj_sf_n"],
    )

    return GetLitterModelReturnData(
        litter_external_base=litter_external_base,
        litter_external_project=litter_external_project,
        synthetic_fertiliser_base=synthetic_fertiliser_base,
        synthetic_fertiliser_project=synthetic_fertiliser_project,
    )


class GetCropModelReturnData(NamedTuple):
    crop_base: List[CropModel.CropModelData]
    crop_par_base: List[CropParams.CropParamsData]
    crop_project: List[CropModel.CropModelData]
    crop_par_project: List[CropParams.CropParamsData]


def get_crop_model_data(
    intervention_input: Dict[str, Union[float, int]], no_of_years: int
) -> GetCropModelReturnData:
    crop_base, crop_par_base = CropModel.get_crop_bases(
        input_data=intervention_input,
        no_of_years=no_of_years,
        start_index=1,
        end_index=3,
    )
    crop_project, crop_par_project = CropModel.get_crop_projects(
        input_data=intervention_input,
        no_of_years=no_of_years,
        start_index=1,
        end_index=3,
    )

    return GetCropModelReturnData(
        crop_base=crop_base,
        crop_par_base=crop_par_base,
        crop_project=crop_project,
        crop_par_project=crop_par_project,
    )


class GetSoilCarbonReturnData(NamedTuple):
    base_forward_soil_data: ForwardSoilModelData
    project_forward_soil_data: ForwardSoilModelData
    for_soil: ForwardSoilModelData


def get_soil_carbon_data(
    no_of_years: int,
    climate: Climate.ClimateData,
    soil: SoilParams.SoilParamsData,
    inverse_soil_model: InverseSoilModelData,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
    cover_base: np.ndarray,
    cover_proj: np.ndarray,
    create_forward_soil_model,
    create_inverse_soil_model,
) -> GetSoilCarbonReturnData:
    # Solve to y=0
    for_soil = create_forward_soil_model(
        soil,
        climate,
        cover_base,
        no_of_years=no_of_years,
        Ci=inverse_soil_model.eq_C,
        crop=crop_base,
        fire=fire_base,
        solve_to_value=True,
    )

    # Soil carbon for baseline and project
    base_forward_soil_data = create_forward_soil_model(
        soil=soil,
        climate=climate,
        cover=cover_base,
        Ci=for_soil.SOC[-1],
        no_of_years=no_of_years,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fire=fire_base,
    )

    project_forward_soil_data = create_forward_soil_model(
        soil,
        climate,
        cover_proj,
        Ci=for_soil.SOC[-1],
        no_of_years=no_of_years,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fire=fire_project,
    )

    return GetSoilCarbonReturnData(
        base_forward_soil_data=base_forward_soil_data,
        project_forward_soil_data=project_forward_soil_data,
        for_soil=for_soil,
    )


class GetEmissionsReturnData(NamedTuple):
    emit_base_emissions: np.ndarray
    emit_project_emissions: np.ndarray


def get_emissions_data(
    no_of_years: int,
    base_forward_soil_data: ForwardSoilModelData,
    project_forward_soil_data: ForwardSoilModelData,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
    synthetic_fertiliser_base: LitterModel.LitterModelData,
    synthetic_fertiliser_project: LitterModel.LitterModelData,
    fire_base: np.ndarray,
    fire_off_base: bool,
    fire_project: np.ndarray,
    fire_off_project: bool,
    gwp: dict,
) -> GetEmissionsReturnData:
    # Emissions stuff
    emit_base_emissions = Emit.create(
        no_of_years=no_of_years,
        forward_soil_model=base_forward_soil_data,
        crop=crop_base,
        tree=[tree_base],
        litter=[litter_external_base],
        fert=[synthetic_fertiliser_base],
        fire=fire_base,
        burn_off=fire_off_base,
        gwp=gwp,
    )
    emit_project_emissions = Emit.create(
        no_of_years=no_of_years,
        forward_soil_model=project_forward_soil_data,
        crop=crop_project,
        tree=tree_projects,
        litter=[litter_external_project],
        fert=[synthetic_fertiliser_project],
        fire=fire_project,
        burn_off=fire_off_project,
        gwp=gwp,
    )

    return GetEmissionsReturnData(
        emit_base_emissions=emit_base_emissions,
        emit_project_emissions=emit_project_emissions,
    )


class GetEmissionsWithDifferenceReturnData(NamedTuple):
    base_emissions: np.ndarray
    project_emissions: np.ndarray
    difference: np.ndarray


def get_crop_emissions(
    no_of_years: int,
    crop_base: List[CropModel.CropModelData],
    crop_project: List[CropModel.CropModelData],
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    burn_off_base: bool,
    burn_off_project: bool,
    gwp: dict,
) -> GetEmissionsWithDifferenceReturnData:
    crop_base_emissions = Emit.create(
        no_of_years=no_of_years,
        crop=crop_base,
        fire=fire_base,
        gwp=gwp,
        burn_off=burn_off_base,
    )
    crop_project_emissions = Emit.create(
        no_of_years=no_of_years,
        crop=crop_project,
        fire=fire_project,
        gwp=gwp,
        burn_off=burn_off_project,
    )
    crop_difference = crop_project_emissions - crop_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=crop_base_emissions,
        project_emissions=crop_project_emissions,
        difference=crop_difference,
    )


def get_fertiliser_emissions(
    no_of_years: int,
    synthetic_fertiliser_base: LitterModel.LitterModelData,
    synthetic_fertiliser_project: LitterModel.LitterModelData,
    gwp: dict,
) -> GetEmissionsWithDifferenceReturnData:
    fertiliser_base_emissions = Emit.create(
        no_of_years=no_of_years,
        fert=[synthetic_fertiliser_base],
        gwp=gwp,
    )
    fertiliser_project_emissions = Emit.create(
        no_of_years=no_of_years,
        fert=[synthetic_fertiliser_project],
        gwp=gwp,
    )
    fertiliser_difference = fertiliser_project_emissions - fertiliser_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=fertiliser_base_emissions,
        project_emissions=fertiliser_project_emissions,
        difference=fertiliser_difference,
    )


def get_litter_emissions(
    no_of_years: int,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    litter_external_base: LitterModel.LitterModelData,
    litter_external_project: LitterModel.LitterModelData,
    gwp: dict,
) -> GetEmissionsWithDifferenceReturnData:
    litter_base_emissions = Emit.create(
        no_of_years=no_of_years,
        litter=[litter_external_base],
        fire=fire_base,
        gwp=gwp,
    )
    litter_project_emissions = Emit.create(
        no_of_years=no_of_years,
        litter=[litter_external_project],
        fire=fire_project,
        gwp=gwp,
    )
    litter_difference = litter_project_emissions - litter_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=litter_base_emissions,
        project_emissions=litter_project_emissions,
        difference=litter_difference,
    )


def get_fire_emissions(
    no_of_years: int,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    gwp: dict,
) -> GetEmissionsWithDifferenceReturnData:
    fire_base_emissions = Emit.create(
        no_of_years=no_of_years,
        fire=fire_base,
        gwp=gwp,
    )
    fire_project_emissions = Emit.create(
        no_of_years=no_of_years,
        fire=fire_project,
        gwp=gwp,
    )
    fire_difference = fire_project_emissions - fire_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=fire_base_emissions,
        project_emissions=fire_project_emissions,
        difference=fire_difference,
    )


def get_tree_emissions(
    no_of_years: int,
    fire_base: np.ndarray,
    fire_project: np.ndarray,
    tree_base: TreeModel.TreeModel,
    tree_projects: List[TreeModel.TreeModel],
    gwp: dict,
) -> GetEmissionsWithDifferenceReturnData:
    tree_base_emissions = Emit.create(
        no_of_years=no_of_years,
        tree=[tree_base],
        fire=fire_base,
        gwp=gwp,
    )
    tree_project_emissions = Emit.create(
        no_of_years=no_of_years,
        tree=tree_projects,
        fire=fire_project,
        gwp=gwp,
    )
    tree_difference = tree_project_emissions - tree_base_emissions

    return GetEmissionsWithDifferenceReturnData(
        base_emissions=tree_base_emissions,
        project_emissions=tree_project_emissions,
        difference=tree_difference,
    )


class InterventionReturnData(NamedTuple):
    soil_base_emissions: np.ndarray
    soil_project_emissions: np.ndarray
    soil_difference: np.ndarray
    tree_base_emissions: np.ndarray
    tree_project_emissions: np.ndarray
    tree_difference: np.ndarray
    fire_base_emissions: np.ndarray
    fire_project_emissions: np.ndarray
    fire_difference: np.ndarray
    litter_base_emissions: np.ndarray
    litter_project_emissions: np.ndarray
    litter_difference: np.ndarray
    fertiliser_base_emissions: np.ndarray
    fertiliser_project_emissions: np.ndarray
    fertiliser_difference: np.ndarray
    crop_base_emissions: np.ndarray
    crop_project_emissions: np.ndarray
    crop_difference: np.ndarray
    soil: SoilParams.SoilParamsData
    climate: Climate.ClimateData
    tree_growths: List[TreeGrowth.TreeGrowth]
    tree_projects: List[TreeModel.TreeModel]
    crop_base: List[CropModel.CropModelData]
    crop_project: List[CropModel.CropModelData]
    crop_par_base: List[CropParams.CropParamsData]
    crop_par_project: List[CropParams.CropParamsData]
    emit_base_emissions: np.ndarray
    emit_project_emissions: np.ndarray
    for_soil: ForwardSoilModelData
    base_forward_soil_data: ForwardSoilModelData
    project_forward_soil_data: ForwardSoilModelData
    inverse_soil_model: InverseSoilModelData


def handle_intervention(
    intervention_input: Dict[str, Union[float, int]],
    create_forward_soil_model,
    create_inverse_soil_model,
    n_cohorts: int,
    plot_index: int,
    allometry: str = CONSTANTS.DEFAULT_ALLOMORPHY,
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
    use_api: bool = CONSTANTS.DEFAULT_USE_API,
):
    no_of_years = get_int(CONSTANTS.NO_OF_YEARS_KEY, intervention_input)
    plot_id = get_int("plot_name", intervention_input)

    # ----------
    # LOCATION INFORMATION
    # ----------
    location = get_location(intervention_input)
    climate = Climate.from_location(location, use_api=use_api)

    # ----------
    # SOIL EQUILIBRIUM SOLVE
    # ----------
    soil = SoilParams.get_soil_params(
        location=location, use_api=use_api, plot_index=plot_index, plot_id=plot_id
    )

    # inverse uses one monthly vector of climate data, but climate is a 12 * years vector, so average:
    inverse_climate = deepcopy(climate)
    inverse_climate.evaporation = np.mean(inverse_climate.evaporation.reshape(-1, 12), axis=0)
    inverse_climate.temperature = np.mean(inverse_climate.temperature.reshape(-1, 12), axis=0)
    inverse_climate.rain = np.mean(inverse_climate.rain.reshape(-1, 12), axis=0)

    inverse_soil_model = create_inverse_soil_model(soil, inverse_climate)

    # ----------
    # MODEL DATA
    # ----------
    crop_model_data = get_crop_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
    )

    fire_model_data = get_fire_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
    )

    litter_model_data = get_litter_model_data(
        no_of_years=no_of_years, intervention_input=intervention_input
    )

    tree_model_data = get_tree_model_data(
        no_of_years=no_of_years,
        intervention_input=intervention_input,
        no_of_cohorts=n_cohorts,
        allometry=allometry
    )

    # ----------
    # EMISSIONS
    # ----------
    crop_emissions = get_crop_emissions(
        no_of_years=no_of_years,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        burn_off_base=fire_model_data.fire_off_base,
        burn_off_project=fire_model_data.fire_off_proj,
        gwp=gwp,
    )

    fertiliser_emissions = get_fertiliser_emissions(
        no_of_years=no_of_years,
        synthetic_fertiliser_base=litter_model_data.synthetic_fertiliser_base,
        synthetic_fertiliser_project=litter_model_data.synthetic_fertiliser_project,
        gwp=gwp,
    )

    litter_emissions = get_litter_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
        gwp=gwp,
    )

    fire_emissions = get_fire_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        gwp=gwp,
    )

    tree_emissions = get_tree_emissions(
        no_of_years=no_of_years,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
        gwp=gwp,
    )

    # ----------
    # SOIL EMISSIONS
    # ----------
    soil_carbon_data = get_soil_carbon_data(
        no_of_years=no_of_years,
        climate=climate,
        soil=soil,
        inverse_soil_model=inverse_soil_model,
        fire_base=fire_model_data.fire_base,
        fire_project=fire_model_data.fire_project,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
        cover_base=intervention_input["base_cover"],
        cover_proj=intervention_input["proj_cover"],
        create_forward_soil_model=create_forward_soil_model,
        create_inverse_soil_model=create_inverse_soil_model,
    )

    emissions = get_emissions_data(
        no_of_years=no_of_years,
        base_forward_soil_data=soil_carbon_data.base_forward_soil_data,
        project_forward_soil_data=soil_carbon_data.project_forward_soil_data,
        crop_base=crop_model_data.crop_base,
        crop_project=crop_model_data.crop_project,
        tree_base=tree_model_data.tree_base,
        tree_projects=tree_model_data.tree_projects,
        litter_external_base=litter_model_data.litter_external_base,
        litter_external_project=litter_model_data.litter_external_project,
        synthetic_fertiliser_base=litter_model_data.synthetic_fertiliser_base,
        synthetic_fertiliser_project=litter_model_data.synthetic_fertiliser_project,
        fire_base=fire_model_data.fire_base,
        fire_off_base=fire_model_data.fire_off_base,
        fire_project=fire_model_data.fire_project,
        fire_off_project=fire_model_data.fire_off_proj,
        gwp=gwp,
    )

    soil_base_emissions = emissions.emit_base_emissions - (
        crop_emissions.base_emissions
        + fertiliser_emissions.base_emissions
        + litter_emissions.base_emissions
        + fire_emissions.base_emissions
        + tree_emissions.base_emissions
    )

    soil_project_emissions = emissions.emit_project_emissions - (
        crop_emissions.project_emissions
        + fertiliser_emissions.project_emissions
        + litter_emissions.project_emissions
        + fire_emissions.project_emissions
        + tree_emissions.project_emissions
    )

    soil_difference = soil_project_emissions - soil_base_emissions

    result = InterventionReturnData(
        climate=climate,
        crop_base_emissions=crop_emissions.base_emissions,
        crop_base=crop_model_data.crop_base,
        crop_difference=crop_emissions.difference,
        crop_par_base=crop_model_data.crop_par_base,
        crop_par_project=crop_model_data.crop_par_project,
        crop_project_emissions=crop_emissions.project_emissions,
        crop_project=crop_model_data.crop_project,
        emit_base_emissions=emissions.emit_base_emissions,
        emit_project_emissions=emissions.emit_project_emissions,
        fertiliser_base_emissions=fertiliser_emissions.base_emissions,
        fertiliser_difference=fertiliser_emissions.difference,
        fertiliser_project_emissions=fertiliser_emissions.project_emissions,
        fire_base_emissions=fire_emissions.base_emissions,
        fire_difference=fire_emissions.difference,
        fire_project_emissions=fire_emissions.project_emissions,
        for_soil=soil_carbon_data.for_soil,
        inverse_soil_model=inverse_soil_model,
        litter_base_emissions=litter_emissions.base_emissions,
        litter_difference=litter_emissions.difference,
        litter_project_emissions=litter_emissions.project_emissions,
        base_forward_soil_data=soil_carbon_data.base_forward_soil_data,
        project_forward_soil_data=soil_carbon_data.project_forward_soil_data,
        soil_base_emissions=soil_base_emissions,
        soil_difference=soil_difference,
        soil_project_emissions=soil_project_emissions,
        soil=soil,
        tree_base_emissions=tree_emissions.base_emissions,
        tree_difference=tree_emissions.difference,
        tree_growths=tree_model_data.tree_growths,
        tree_project_emissions=tree_emissions.project_emissions,
        tree_projects=tree_model_data.tree_projects,
    )

    return result
