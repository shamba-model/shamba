from typing import Dict, List, NamedTuple, Any, Callable, Optional, Tuple
import csv
import concurrent.futures
import model.soil_params as SoilParams
from model.common.calculate_emissions import handle_intervention
import model.common.constants as CONSTANTS
import model.monte_carlo.sampler as sampler
import numpy as np
from model.emit import EmissionFactors
from model.monte_carlo.model_parameter_distributions import MODEL_PARAMETER_DISTRIBUTIONS
from model.monte_carlo.distribution_handler import DistributionSpec


class SampleArgs(NamedTuple):
    perturbed_intervention_input: Dict[str, Any]
    create_forward_soil_model: Callable
    create_inverse_soil_model: Callable
    n_cohorts: int
    plot_index: int
    soil_params: Optional[SoilParams.SoilParamsData] = None
    emission_factors: EmissionFactors = EmissionFactors()
    allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP]
    use_api: bool = CONSTANTS.DEFAULT_USE_API


def _run_single_sample(arguments: SampleArgs):
    return handle_intervention(
        intervention_input=arguments.perturbed_intervention_input,
        create_forward_soil_model=arguments.create_forward_soil_model,
        create_inverse_soil_model=arguments.create_inverse_soil_model,
        n_cohorts=arguments.n_cohorts,
        plot_index=arguments.plot_index,
        soil_override=arguments.soil_params,
        allometry=arguments.allometry,
        gwp=arguments.gwp,
        use_api=arguments.use_api,
        emission_factors=arguments.emission_factors
    )


def run_monte_carlo(
    base_input_dict: Dict[str, Any],
    soil_params: SoilParams.SoilParamsData,
    climate,
    n_samples: int,
    create_forward_soil_model: Callable,
    create_inverse_soil_model: Callable,
    n_cohorts: int,
    plot_index: int,
    sample_emission_factors: bool = False,
    distribution_dict: Optional[Dict] = None,
    model_params: Optional[EmissionFactors] = EmissionFactors(),
    emission_distribution_dict: Dict[str, DistributionSpec] = MODEL_PARAMETER_DISTRIBUTIONS,
    allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY,
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
    use_api: bool = CONSTANTS.DEFAULT_USE_API,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:

    rng = np.random.default_rng(seed)

    soil_samples = sampler.sample_soil_params(
        soil=soil_params,
        n_samples=n_samples,
        rng=rng,
    )

    climate_samples = sampler.sample_climate_params(
        climate=climate,
        n_samples=n_samples,
        rng=rng,
    )

    if sample_emission_factors:
        emission_factor_samples = sampler.sample_model_params(
            n_samples=n_samples,
            rng=rng,
            base_model_params=model_params,
            distribution_dict=emission_distribution_dict
        )
    else:
        emission_factor_samples = [EmissionFactors() for _ in range(n_samples)]


    if distribution_dict is None:
        samples = [dict(base_input_dict) for _ in range(n_samples)]
    else:
        samples = sampler.draw_samples(
            base_input_dict=base_input_dict,
            distributions=distribution_dict,
            n_samples=n_samples,
            rng=rng,
        )

    for i in range(n_samples):
        samples[i].update(climate_samples[i])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_single_sample, [
            SampleArgs(
                perturbed_intervention_input=samples[i],
                emission_factors = emission_factor_samples[i],
                create_forward_soil_model=create_forward_soil_model,
                create_inverse_soil_model=create_inverse_soil_model,
                n_cohorts=n_cohorts,
                plot_index=plot_index,
                soil_params=soil_samples[i],
                allometry=allometry,
                gwp=gwp,
                use_api=use_api,
            )
            for i in range(n_samples)
        ]))

    return results


_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

_BASE_GETTERS = {
    "soil":       lambda r: r.soil_base_emissions,
    "tree":       lambda r: r.tree_base_emissions,
    "fire":       lambda r: r.fire_base_emissions,
    "litter":     lambda r: r.litter_base_emissions,
    "fertiliser": lambda r: r.fertiliser_base_emissions,
    "crop":       lambda r: r.crop_base_emissions,
    "emit":       lambda r: r.emit_base_emissions,
}

_PROJ_GETTERS = {
    "soil":       lambda r: r.soil_project_emissions,
    "tree":       lambda r: r.tree_project_emissions,
    "fire":       lambda r: r.fire_project_emissions,
    "litter":     lambda r: r.litter_project_emissions,
    "fertiliser": lambda r: r.fertiliser_project_emissions,
    "crop":       lambda r: r.crop_project_emissions,
    "emit":       lambda r: r.emit_project_emissions,
}

_DIFF_GETTERS = {
    "soil":       lambda r: r.soil_difference,
    "tree":       lambda r: r.tree_difference,
    "fire":       lambda r: r.fire_difference,
    "litter":     lambda r: r.litter_difference,
    "fertiliser": lambda r: r.fertiliser_difference,
    "crop":       lambda r: r.crop_difference,
    "emit":       lambda r: r.emit_project_emissions - r.emit_base_emissions,
}


class MCSummaries(NamedTuple):
    base: Dict[str, np.ndarray]
    project: Dict[str, np.ndarray]
    diff: Dict[str, np.ndarray]


def _compute_summary(
    mc_results: List[Any],
    getters: Dict[str, Any],
    quantiles: Tuple[float, ...],
) -> Dict[str, np.ndarray]:
    summary = {}
    for pool_name, getter in getters.items():
        data = np.array([getter(r) for r in mc_results])  # gathers n_samples arrays of results length n_years, array: (n_samples, n_years)
        summary[f"{pool_name}_mean"] = data.mean(axis=0) # array: (n_years,)
        summary[f"{pool_name}_std"] = data.std(axis=0)
        for q in quantiles:
            label = f"q{int(round(q * 100)):02d}"
            summary[f"{pool_name}_{label}"] = np.quantile(data, q, axis=0)
    return summary


def summarise_mc_results(
    mc_results: List[Any],
    quantiles: Tuple[float, ...] = _QUANTILES,
) -> MCSummaries:
    """Reduce a list of InterventionReturnData to three per-year summary dicts.

    Each dict has the same shape: columns {pool}_mean, {pool}_std,
    {pool}_q05, {pool}_q25, {pool}_q50, {pool}_q75, {pool}_q95 for pools
    soil, tree, fire, litter, fertiliser, crop, emit. Values represent
    baseline emissions, project emissions, and their difference respectively.
    """
    return MCSummaries(
        base=_compute_summary(mc_results, _BASE_GETTERS, quantiles),
        project=_compute_summary(mc_results, _PROJ_GETTERS, quantiles),
        diff=_compute_summary(mc_results, _DIFF_GETTERS, quantiles),
    )


def write_mc_summary_csv(summary: Dict[str, np.ndarray], output_path: str) -> None:
    """Write a summary dict to a CSV file, one row per year."""
    cols = list(summary.keys())
    n_years = len(next(iter(summary.values())))
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year"] + cols)
        for year in range(1, n_years + 1):
            writer.writerow([year] + [float(summary[col][year - 1]) for col in cols])
