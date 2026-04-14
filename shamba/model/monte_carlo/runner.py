from typing import Dict, List, NamedTuple, Any, Callable, Optional

import concurrent.futures
import model.soil_params as SoilParams
from model.common.calculate_emissions import handle_intervention
import model.common.constants as CONSTANTS
import model.monte_carlo.sampler as sampler
import numpy as np

class SampleArgs(NamedTuple):
    perturbed_intervention_input: Dict[str, Any]
    create_forward_soil_model: Callable
    create_inverse_soil_model: Callable
    n_cohorts: int
    plot_index: int
    soil_params: Optional[SoilParams.SoilParamsData] = None
    allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP]
    use_api: bool = CONSTANTS.DEFAULT_USE_API

def _run_single_sample(
arguments: SampleArgs
):
    return handle_intervention(
        intervention_input = arguments.perturbed_intervention_input,
        create_forward_soil_model=arguments.create_forward_soil_model,
        create_inverse_soil_model=arguments.create_inverse_soil_model,
        n_cohorts=arguments.n_cohorts,
        plot_index=arguments.plot_index,
        soil_override=arguments.soil_params,
        allometry=arguments.allometry,
        gwp=arguments.gwp,
        use_api=arguments.use_api,
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
        distribution_dict: Optional[Dict] = None,
        allometry: List[str] = CONSTANTS.DEFAULT_ALLOMORPHY,
        gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
        use_api: bool = CONSTANTS.DEFAULT_USE_API,
        seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
  
  rng = np.random.default_rng(seed)
  
  soil_samples = sampler.sample_soil_params(
      soil=soil_params,
      n_samples=n_samples,
      rng=rng
  )

  climate_samples = sampler.sample_climate_params(
      climate=climate,
      n_samples=n_samples,
      rng=rng)
  
  if distribution_dict is None:
     samples = [dict(base_input_dict) for _ in range(n_samples)]

  else:
     samples = sampler.draw_samples(
        base_input_dict=base_input_dict,
        distributions=distribution_dict,
        n_samples=n_samples,
        seed=seed
     )

  for i in range(n_samples):
        samples[i].update(climate_samples[i])

  with concurrent.futures.ProcessPoolExecutor() as executor:
     results = list(executor.map(_run_single_sample, [SampleArgs(
            perturbed_intervention_input=samples[i],
            create_forward_soil_model=create_forward_soil_model,
            create_inverse_soil_model=create_inverse_soil_model,
            n_cohorts=n_cohorts,
            plot_index=plot_index,
            soil_params=soil_samples[i],
            allometry=allometry,
            gwp=gwp,
            use_api=use_api
        ) for i in range(n_samples)]))

  return results