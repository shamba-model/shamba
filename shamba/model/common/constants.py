from datetime import datetime

"""
This file contains constants used in SHAMBA. 
# Model constants #
- must not be changed
# Emission factors #
- should be changed if user has better data for their context
# Default values #
- must be changed if not appropriate for user's context
# Example project configuration #
- values used in example project provided, do not change
# Project data keys #
- strings used in the csv input, do not change
"""

# =============================================================================
#  Model constants
# =============================================================================

# Global warming potential from IPCC Assessment Reports:

GWP_SAR = {"N2O": 310, "CH4": 21}
GWP_AR4 = {"N2O": 298, "CH4": 25}
GWP_AR5 = {"N2O": 265, "CH4": 28}
GWP_AR6 = {"N2O": 273, "CH4": 27}

GWP_list = {
    "GWP SAR (1995)": GWP_SAR,
    "GWP AR4 (2007)": GWP_AR4,
    "GWP AR5 (2013)": GWP_AR5,
    "GWP AR6 (2021)": GWP_AR6,
}

C_to_CO2_conversion_factor = 44.0 / 12  # for CO2-C to CO2
N_to_N2O_conversion_factor = 44.0 / 28  # for N2O-N to N2O

# =============================================================================
#  Emission factors
# =============================================================================

# From Table 2.5 IPCC 2006 GHG Inventory (unchanged in 2019)
# "EMISSION FACTORS (g kg-1 DRY MATTER BURNT) FOR VARIOUS TYPES OF BURNING."
# Tree based on "Tropical Forest".
ef_burn_default = {"crop_N2O": 0.07, "crop_CH4": 2.7, "tree_N2O": 0.2, "tree_CH4": 6.8}

# From Table 11.1 IPCC 2019 GHG Inventory
#  "DEFAULT EMISSION FACTORS TO ESTIMATE DIRECT N2O EMISSIONS FROM MANAGED SOILS"
# (EF_1)
ef_N_inputs_default = 0.01  # [kg N20-N/kg N]

# From Table 2.6 IPCC 2019 GHG Inventory
# "COMBUSTION FACTOR VALUES (PROPORTION OF PREFIRE FUEL BIOMASS CONSUMED)
# FOR FIRES IN A RANGE OF VEGETATION TYPES"
# Tree based on "Savanna Woodlands", crop based on "Other Crops"
combustion_factor_default = {"crop": 0.85, "tree": 0.74}

# From Table 11.3 IPCC 2019 GHG Inventory
# "DEFAULT EMISSION, VOLATILISATION AND LEACHING FACTORS FOR
# INDIRECT SOIL N2O EMISSIONS"
# (Frac_gasf, Frac_gasm)
volatile_frac_synthetic_fertiliser_default = 0.11  # [(kg NH3-N + NOx-N) / kg N applied]
volatile_frac_organic_fertiliser_default = 0.21  # [(kg NH3-N + NOx-N) / kg N applied]

# =============================================================================
#  Default values
# =============================================================================

# -------------------------
# a) values overwritten by the user's input when shamba_command_line.py is run:

DEFAULT_USE_CLIMATE_API = True
DEFAULT_USE_SOIL_API = True
DEFAULT_ALLOMORPHY = "chave dry"

# -------------------------
# b) values with a general default, that users may want to manually change:

DEFAULT_GWP = "GWP AR6 (2021)"
TREE_ROOT_IN_TOP_30 = 0.7  # %
CROP_ROOT_IN_TOP_30 = 0.7  # %
ORGANIC_INPUT_N = 0.018  # kg N/ kg input
ORGANIC_INPUT_C = 0.5  # kg C/kg input
# -------------------------

# =============================================================================
#  Example project configuration
# =============================================================================

EXAMPLE_N_COHORTS = 3
EXAMPLE_TREE_STANDARD_DENSITY = 100  # trees / ha
EXAMPLE_NO_OF_YEARS = 10

# =============================================================================
#  Project data keys
# =============================================================================

LOCATION_LATITIUDE_KEY = "lat"
LOCATION_LONGITUDE_KEY = "lon"
NO_OF_YEARS_KEY = "yrs_proj"
NO_OF_COHORTS_KEY = "no_of_cohorts"
ALLOMETRY_KEY = "allometry"
# ----------
SPECIES_BASE_KEY = "species_base"
SPECIES_1_KEY = "species1"
BASE_PLANT_YEAR_KEY = "base_plant_yr"
# ----------
THINNING_BASE_YEAR_1_KEY = "thin_base_yr1"
THINNING_BASE_YEAR_2_KEY = "thin_base_yr2"
# ----------
THINNING_BASE_PC1_KEY = "thin_base_pc1"
THINNING_BASE_PC2_KEY = "thin_base_pc2"
# ----------
THINNING_PROJECT_YEAR_1_KEY = "thin_proj_yr1"
THINNING_PROJECT_YEAR_2_KEY = "thin_proj_yr2"
THINNING_PROJECT_YEAR_3_KEY = "thin_proj_yr3"
THINNING_PROJECT_YEAR_4_KEY = "thin_proj_yr4"
# ----------
THINNING_PROJECT_PC1_KEY = "thin_proj_pc1"
THINNING_PROJECT_PC2_KEY = "thin_proj_pc2"
THINNING_PROJECT_PC3_KEY = "thin_proj_pc3"
THINNING_PROJECT_PC4_KEY = "thin_proj_pc4"
# ----------
THINNING_BASE_BR_KEY = "thin_base_br"
THINNING_BASE_ST_KEY = "thin_base_st"
# ----------
THINNING_PROJECT_BR_KEY = "thin_proj_br"
THINNING_PROJECT_ST_KEY = "thin_proj_st"
# ----------
BASE_MORTALITY_KEY = "base_mort"
PROJECT_MORTALITY_KEY = "proj_mort"
# ----------
MORTALITY_BASE_BR_KEY = "mort_base_br"
MORTALITY_BASE_ST_KEY = "mort_base_st"
# ----------
MORTALITY_PROJECT_BR_KEY = "mort_proj_br"
MORTALITY_PROJECT_ST_KEY = "mort_proj_st"
# ----------
BASE_PLANT_DENSITY_KEY = "base_plant_dens"
# ----------
FIRE_INTERVAL_BASE_KEY = "fire_int_base"
FIRE_PRES_BASE_KEY = "fire_pres_base"
# ----------
FIRE_INTERVAL_PROJECT_KEY = "fire_int_proj"
FIRE_PRES_PROJECT_KEY = "fire_pres_proj"
# ----------
BASE_LITTER_INTERVAL_KEY = "base_lit_int"
BASE_LITTER_QUANTITY_KEY = "base_lit_qty1"
# ----------
PROJECT_LITTER_INTERVAL_KEY = "proj_lit_int"
PROJECT_LITTER_QUANTITY_KEY = "proj_lit_qty1"
# ----------
BASE_SYNTHETIC_FERTILISER_INTERVAL_KEY = "base_sf_int"
BASE_SYNTHETIC_FERTILISER_QUANTITY_KEY = "base_sf_qty1"
BASE_SYNTHETIC_FERTILISER_N_KEY = "base_sf_n1"
# ----------
PROJECT_SYNTHETIC_FERTILISER_INTERVAL_KEY = "proj_sf_int"
PROJECT_SYNTHETIC_FERTILISER_QUANTITY_KEY = "proj_sf_qty1"
PROJECT_SYNTHETIC_FERTILISER_N_KEY = "proj_sf_n1"
# ----------
BASE_COVER_MONTH_START_KEY = "base_cvr_mth_st"
BASE_COVER_MONTH_END_KEY = "base_cvr_mth_en"
BASE_COVER_PRES_KEY = "base_cvr_pres"
# ----------
PROJECT_COVER_MONTH_START_KEY = "proj_cvr_mth_st"
PROJECT_COVER_MONTH_END_KEY = "proj_cvr_mth_en"
PROJECT_COVER_PRES_KEY = "proj_cvr_pres"
# ---------------------------------------------
