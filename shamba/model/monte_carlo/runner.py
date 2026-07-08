from datetime import datetime, timezone
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
from model.climate import ClimateData

class MCSummaries(NamedTuple):
    base: Dict[str, np.ndarray]
    project: Dict[str, np.ndarray]
    diff: Dict[str, np.ndarray]

class SampleArgs(NamedTuple):
    perturbed_intervention_input: Dict[str, Any]
    create_forward_soil_model: Callable
    create_inverse_soil_model: Callable
    n_proj_cohorts: int
    n_base_cohorts: int
    plot_index: int
    soil_params: SoilParams.SoilParamsData
    climate: ClimateData
    tree_species_data: Dict[int, Dict]
    crop_species_data: Dict[int, Dict]
    pool_species_data: Dict[int, Dict]
    emission_factors: EmissionFactors = EmissionFactors()
    allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP]



def _run_single_sample(arguments: SampleArgs):
    return handle_intervention(
        intervention_input=arguments.perturbed_intervention_input,
        climate = arguments.climate,
        soil = arguments.soil_params,
        create_forward_soil_model=arguments.create_forward_soil_model,
        create_inverse_soil_model=arguments.create_inverse_soil_model,
        n_proj_cohorts=arguments.n_proj_cohorts,
        n_base_cohorts=arguments.n_base_cohorts,
        plot_index=arguments.plot_index,
        allometry=arguments.allometry,
        gwp=arguments.gwp,
        emission_factors=arguments.emission_factors,
        tree_species_data=arguments.tree_species_data,
        crop_species_data=arguments.crop_species_data,
        pool_species_data=arguments.pool_species_data,
    )


def run_monte_carlo(
    base_input_dict: Dict[str, Any],
    soil_params: SoilParams.SoilParamsData,
    climate,
    n_samples: int,
    create_forward_soil_model: Callable,
    create_inverse_soil_model: Callable,
    n_proj_cohorts: int,
    n_base_cohorts: int,
    plot_index: int,
    tree_species_data: Dict[int, Dict],
    crop_species_data: Dict[int, Dict],
    pool_species_data: Dict[int, Dict],
    sample_emission_factors: bool = False,
    distribution_dict: Optional[Dict] = None,
    model_params: Optional[EmissionFactors] = EmissionFactors(),
    emission_distribution_dict: Dict[str, DistributionSpec] = MODEL_PARAMETER_DISTRIBUTIONS,
    allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY,
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
    seed: Optional[int] = None,
    checkpoint_every: int = 0,
    on_checkpoint: Optional[Callable[[int, MCSummaries], None]] = None,
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

    if checkpoint_every > 0:
        batch_starts = range(0, n_samples, checkpoint_every)
        batch_size = checkpoint_every
    else:
        batch_starts = [0]
        batch_size = n_samples

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for start in batch_starts:
            stop = start + batch_size
            samples_batch = samples[start:stop]
            emission_factor_samples_batch = emission_factor_samples[start:stop]
            soil_samples_batch = soil_samples[start:stop]
            results.extend(list(executor.map(_run_single_sample, [
                SampleArgs(
                    perturbed_intervention_input=samples_batch[i],
                    emission_factors=emission_factor_samples_batch[i],
                    create_forward_soil_model=create_forward_soil_model,
                    create_inverse_soil_model=create_inverse_soil_model,
                    n_proj_cohorts=n_proj_cohorts,
                    n_base_cohorts=n_base_cohorts,
                    plot_index=plot_index,
                    soil_params=soil_samples_batch[i],
                    climate=climate_samples[i],
                    allometry=allometry,
                    gwp=gwp,
                    tree_species_data=tree_species_data,
                    crop_species_data=crop_species_data,
                    pool_species_data=pool_species_data,
                )
                for i in range(len(samples_batch))
            ])))
            if on_checkpoint:
                on_checkpoint(len(results), summarise_mc_results(results))

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


def write_mc_metadata(
    output_path: str,
    n_samples: int,
    seed: Optional[int],
    soil_params: SoilParams.SoilParamsData,
    climate,
    distribution_dict: Optional[Dict[str, DistributionSpec]],
    sample_emission_factors: bool,
) -> None:
    """Write a plain-text summary of what the Monte Carlo run sampled."""
    lines = []
    lines.append("SHAMBA Monte Carlo run metadata")
    lines.append("=" * 40)
    lines.append(f"Run timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Samples       : {n_samples}")
    if seed is None:
        lines.append("Seed          : not set — run is not reproducible")
    else:
        lines.append(f"Seed          : {seed}")

    lines.append("")
    lines.append("Parameters always sampled")
    lines.append("-" * 30)

    # Soil uncertainty (sigma inferred from Q0.05/Q0.95, matching sampler logic)
    cy0_sigma = (soil_params.Cy0_q95 - soil_params.Cy0_q05) / (2.0 * 1.645)
    clay_sigma = (soil_params.clay_q95 - soil_params.clay_q05) / (2.0 * 1.645)
    lines.append(f"  Soil Cy0  : mean={soil_params.Cy0:.4g}, sigma={cy0_sigma:.4g}"
                 + (" (no uncertainty — fixed value)" if cy0_sigma == 0.0 else ""))
    lines.append(f"  Soil clay : mean={soil_params.clay:.4g}%, sigma={clay_sigma:.4g}"
                 + (" (no uncertainty — fixed value)" if clay_sigma == 0.0 else ""))

    temp_std_mean = float(np.mean(climate.temperature_std))
    rain_std_mean = float(np.mean(climate.rain_std))
    evap_std_mean = float(np.mean(climate.evaporation_std))
    lines.append(f"  Climate Temp : mean monthly std={temp_std_mean:.4g}"
                 + (" (no uncertainty — fixed value)" if temp_std_mean == 0.0 else ""))
    lines.append(f"  Climate Rain : mean monthly std={rain_std_mean:.4g}"
                 + (" (no uncertainty — fixed value)" if rain_std_mean == 0.0 else ""))
    lines.append(f"  Climate evap : mean monthly std={evap_std_mean:.4g}"
                 + (" (no uncertainty — fixed value)" if evap_std_mean == 0.0 else ""))

    lines.append("")
    lines.append("Emission factors sampled")
    lines.append("-" * 30)
    lines.append(f"  {'Yes' if sample_emission_factors else 'No'}")

    lines.append("")
    lines.append("User-defined input distributions")
    lines.append("-" * 30)
    if not distribution_dict:
        lines.append("  None — no distributions file supplied.")
    else:
        header = f"  {'Parameter':<30} {'Distribution':<20} {'Spread lower':>14} {'Spread upper':>14} {'Min abs':>10}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for param, spec in distribution_dict.items():
            min_abs_str = f"{spec.min_abs:.4g}" if spec.min_abs is not None else "—"
            lines.append(
                f"  {param:<30} {spec.distribution:<20} {spec.spread_lower:>14.4g}"
                f" {spec.spread_upper:>14.4g} {min_abs_str:>10}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
