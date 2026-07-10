from typing import NamedTuple, Union

class ExampleSoilModelParams(NamedTuple):
    """Placeholder parameters for the example soil model.

    The example soil model has no working implementation yet (forward_example.py/
    inverse_example.py always fail schema validation). This class has no fields —
    it exists so callers that don't know which soil model is active (e.g.
    handle_intervention()) can pass a soil_model_params object uniformly, matching
    RothCParams' role for the RothC model.
    """
    pass


class RothCParams(NamedTuple):
    """RothC rate-modifier and DPM/RPM partitioning constants.

    Defaults match the values previously hardcoded in roth_c.py, forward_roth_c.py,
    and inverse_roth_c.py — constructing RothCParams() with no arguments is
    numerically equivalent to floating-point precision. The moisture RMF floor is
    deliberately derived as (1 - moisture_b_slope) rather than stored as its own
    field, since floor and slope describe the same linear interpolation and must
    stay coupled under future perturbation (see get_rmf() in roth_c.py) — this
    introduces a single last-bit (~1e-16 relative) rounding difference from the
    prior hardcoded 0.2, well below the model's own numeric tolerances. Exists so
    these constants can be overridden per call (e.g. by Monte Carlo sampling)
    instead of being fixed at import time.
    """
    dpm_frac_crop: float = 0.59       # DPM fraction of crop carbon input
    dpm_frac_tree: float = 0.20       # DPM fraction of tree carbon input
    temp_a1: float = 47.91            # temperature modifier numerator
    temp_a2: float = 106.06           # temperature modifier exponent numerator
    temp_a3: float = 18.27            # temperature modifier offset
    moisture_b_slope: float = 0.80    # moisture modifier slope (min = 1 - slope)
    cover_c: float = 0.60             # rate modifier when soil is covered (bare = 1.0)


# The concrete parameter type depends on which soil model create_forward_soil_model/
# create_inverse_soil_model actually resolve to at runtime — one member per soil model.
SoilModelParams = Union[RothCParams, ExampleSoilModelParams]