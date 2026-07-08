import re
from typing import Dict, Any

import numpy as np
from marshmallow import Schema, fields, post_load

from model.common import csv_handler

# Per-species fields eligible for MC distribution sampling — the single source of
# truth for both the key-matching pattern below and sampler.sample_species_params().
TREE_SPECIES_PARAM_FIELDS = ("wood_dens", "carbon", "root_to_shoot", "nitrogen")

# Keys are prefixed with the catalog name ("tree_") rather than bare field names,
# since some fields (e.g. root_to_shoot) also exist on other species catalogs
# (crop_params.csv) with independently-numbered species codes — a bare key would
# be ambiguous between "tree species 3" and "crop species 3".
# Matches MC distribution keys for per-species tree parameter sampling, e.g.
# "tree_wood_dens_sp2", "tree_carbon_sp1", "tree_nitrogen_sp3".
TREE_SPECIES_DIST_KEY_PATTERN = re.compile(
    rf"^tree_({'|'.join(TREE_SPECIES_PARAM_FIELDS)})_sp(\d+)$"
)


def load_tree_species_data(
    filename: str = "tree_params.csv",
) -> Dict[int, Dict]:
    """Load per-species tree params from csv, keyed by each row's own Sc
    (species code) column — not by row position.

    Each species' display name is read from the file's trailing comment
    column (after wood_dens) — the documented Name column (index 1) just
    repeats the Sc code, so it isn't used as the display name.

    Returns:
        Dict mapping species code (Sc) to its parameter dictionary.
    """
    resolved_path = csv_handler.resolve_csv_path(filename)
    try:
        data = np.atleast_2d(
            np.genfromtxt(resolved_path, skip_header=1, usecols=(0, 2, 3, 4, 5, 6, 7, 8, 9), delimiter=",", comments="#")
        )
        names = np.atleast_1d(
            np.genfromtxt(resolved_path, skip_header=1, usecols=(10,), dtype=str, delimiter=",", comments="#")
        )
    except ValueError as e:
        raise ValueError(
            f"'{filename}' could not be read as a tree params file. It must have "
            f"columns 'Sc,Name,N_leaf,N_branch,N_stem,N_croot,N_Froot,C,rs,wood_dens' "
            f"plus a trailing name/comment column. Original error: {e}"
        ) from e

    if np.isnan(data).any():
        bad_rows = [r + 2 for r in np.where(np.isnan(data).any(axis=1))[0]]
        raise ValueError(
            f"'{filename}' has a missing/blank numeric value in row(s) {bad_rows} "
            f"(counting the header as row 1). Every species row must have a value "
            f"in every column (Sc, N_leaf..N_froot, C, rs, wood_dens)."
        )

    sc_codes = data[:, 0]
    nitrogen = data[:, 1:6]
    carbon = data[:, 6]
    root_to_shoot = data[:, 7]
    wood_density = data[:, 8]

    species_data = {}
    for i, sc in enumerate(sc_codes):
        species = int(sc)
        if species in species_data:
            raise ValueError(
                f"'{filename}' has more than one row with species code (Sc) "
                f"{species} — each species code must appear exactly once."
            )
        species_data[species] = {
            "species": species,
            "name": str(names[i]).strip(),
            "wood_dens": wood_density[i],
            "carbon": carbon[i],
            "nitrogen": nitrogen[i],
            "root_to_shoot": root_to_shoot[i],
        }
    return species_data

class TreeParamsData:
    """
    Object holding tree params.

    Instance variables
    ----------------
    species         tree species code (Sc column in tree_params.csv)
    name            tree species display name
    wood_dens            tree density in g cm^-3
    carbon          tree carbon content as a fraction
    nitrogen        tree nitrogen content as a fraction
    root_to_shoot   tree root-to-shoot ratio
    """

    def __init__(
        self,
        species,
        name,
        wood_dens,
        nitrogen,
        carbon,
        root_to_shoot,
    ):
        self.species = species
        self.name = name
        self.wood_dens = wood_dens
        self.nitrogen = nitrogen
        self.carbon = carbon
        self.root_to_shoot = root_to_shoot


def validate_species(value):
    # Determining whether the value can be interpreted as a string or an integer
    errors = [f"{value} must be convertible to a string."] * (
        not isinstance(str(value), str)
    ) + [f"{value} must be convertible to an integer."] * (
        not value.isdigit() if isinstance(value, str) else False
    )
    return errors


class TreeParamsSchema(Schema):
    species = fields.Raw(required=True, validate=lambda v: validate_species(v))
    name = fields.String(required=True)
    wood_dens = fields.Float(required=True)
    carbon = fields.Float(required=True)
    nitrogen = fields.List(fields.Float, required=True)
    root_to_shoot = fields.Float(required=True)

    @post_load
    def build(self, data, **kwargs):
        return TreeParamsData(**data)


def create(tree_params) -> TreeParamsData:
    """
    Create a TreeParams object from a dict.

    Args: tree_params: dict with tree params

    Returns: TreeParamsData object
    """
    params = {
        "species": tree_params["species"],
        "name": tree_params["name"],
        "wood_dens": tree_params["wood_dens"],
        "carbon": tree_params["carbon"],
        "nitrogen": tree_params["nitrogen"],
        "root_to_shoot": tree_params["root_to_shoot"],
    }

    schema = TreeParamsSchema()
    return schema.load(params)  # type: ignore


def from_species_index(index: int, species_data: Dict[int, Dict]):
    """
    Construct TreeParams from its species code (Sc column in tree_params.csv).

    Args:
        index: species code to look up
        species_data: pre-loaded species data (as returned by
            load_tree_species_data()), loaded once per run by the caller.

    Raises:
        KeyError: if the species code isn't present in tree_params.csv
    """
    index = int(index)
    if index not in species_data:
        raise KeyError(
            f"No tree species with code {index} found in tree_params.csv "
            f"(available codes: {sorted(species_data)})."
        )
    return create(species_data[index])


def save(tree_params: TreeParamsData, file="tree_params.csv"):
    """Save tree params to a csv.
    Default path is in OUTPUT_DIR with filename 'tree_params.csv'

    Args:
        file: name or path to csv file

    """
    data = [
        tree_params.species,
        tree_params.name,
        tree_params.nitrogen[0],
        tree_params.nitrogen[1],
        tree_params.nitrogen[2],
        tree_params.nitrogen[3],
        tree_params.nitrogen[4],
        tree_params.carbon,
        tree_params.root_to_shoot,
        tree_params.wood_dens,
    ]
    cols = [
        "Sc",
        "Name",
        "N_leaf",
        "N_branch",
        "N_stem",
        "N_croot",
        "N_froot",
        "C",
        "rw",
        "wood_dens",
    ]
    csv_handler.print_csv(file, data, col_names=cols)


def create_tree_params_from_species_index(
    csv_input_data: Dict[str, Any], cohort_count: int, species_data: Dict[int, Dict]
):
    tree_params = []

    for i in range(cohort_count):
        key = f"species{i + 1}"
        try:
            species_index = int(csv_input_data[key].item())
            tree_params.append(from_species_index(species_index, species_data=species_data))
        except KeyError:
            raise KeyError(f"Warning: Missing key '{key}' in input data.")
    return tree_params
