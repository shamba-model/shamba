# TODO: Decide whether to retain this module in future releases.
# It exists solely to allow single-row CSV files using the old header naming
# convention to be read without modification. New files should use the new
# naming convention directly (e.g. base_species1, proj_species1, base_plant_yr1).
# If old file support is dropped, this module and its call in
# expand_single_row_data_input can be removed.

def rename_legacy_headers(raw: dict) -> dict:
    """Rename old single-row CSV header names to the new vector-format convention.

    Old format uses:
        species_base, species1/2/3
        base_plant_dens (no cohort index), proj_plant_yr1, proj_plant_dens1

    New format expects:
        base_species1, proj_species1/2/3
        base_plant_dens1, base_plant_yr1, proj_plant_yr1, proj_plant_dens1

    Returns a copy of raw with affected keys renamed.
    """
    renamed = dict(raw)

    if "species_base" in renamed:
        renamed["base_species1"] = renamed.pop("species_base")

    for i in range(1, 4):
        old = f"species{i}"
        if old in renamed:
            renamed[f"proj_species{i}"] = renamed.pop(old)

    if "base_plant_dens" in renamed:
        renamed["base_plant_dens1"] = renamed.pop("base_plant_dens")

    if "base_plant_yr" in renamed:
        renamed["base_plant_yr1"] = renamed.pop("base_plant_yr")

    return renamed
