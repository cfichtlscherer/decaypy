import pandas as pd
import numpy as np
from scipy.constants import Avogadro
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
ALL_NUCLIDE_DATA = DATA_DIR / "data_processed" / "big_df_all_nuclide_data.pickle"

def parse_zamid(zamid) -> tuple[int, int, int]:
    """
    Parse ZAMID encoded as: 10000*Z + 10*A + m
    Returns (Z, A, m).
    """
    s = str(zamid).strip()
    s = s[:-2] if s.endswith(".0") else s  # handle pandas reading as float-like text
    if not s.isdigit():
        raise ValueError(f"Non-numeric ZAMID: {zamid!r}")

    n = int(s)
    m = n % 10
    za = n // 10
    Z = za // 1000
    A = za % 1000
    return Z, A, m


def extract_onix_isotopic_inventory(density_output_onix, volume, step):
    """
    Parse an ONIX `density_output` file and return the isotopic inventory at a
    given time step as absolute number of atoms.

    The ONIX output is read as a whitespace-separated table. The requested `step`
    selects one of the density columns (offset by +2 to account for the leading
    identifier columns). Only rows with a strictly positive density are kept.

    The density values are converted to number of atoms via:
        N_atoms = density * volume * 1e24
    where `volume` is given in cm^3 and the factor 1e24 corresponds to the common
    ONIX convention of densities in atoms/(barn·cm).

    The function also derives nuclide identifiers from the ONIX ZAMID:
      - Z: atomic number
      - A: mass number
      - Excited: excited-state flag encoded in the last digit of ZAMID
    and assigns an excitation energy `Elevel` (MeV) for excited states by looking
    up candidate levels in `big_df_all_nuclide_data.pickle` and selecting the
    smallest positive parent level energy when available.

    Entries flagged as excited (`Excited == 1`) but for which no non-zero energy
    level can be determined (`Elevel == 0`) are written to `excluded_isotopes.csv`
    and removed from the returned inventory.

    Parameters
    ----------
    density_output_onix : str or path-like
        Path to the ONIX density output file.
    volume : float
        Material volume in cm^3 used to convert densities to number of atoms.
    step : int
        ONIX output step index (0-based). Internally shifted by +2 to match the
        column layout of the density output file.

    Returns
    -------
    pandas.DataFrame
        Inventory table with (at least) the columns:
        ["Isotope", "ZAMID", "Number of atoms", "Z", "A", "Excited", "Elevel"].
    """
    densities_panda = pd.read_csv(density_output_onix, sep=r"\s+", header=None, skiprows=7)
    step_col = step + 2  # +2 because we skip the zamid and the initial step column
    selected_columns = [0, 1, step_col]
    condition = densities_panda[step_col] > 0
    number_of_existing_atoms = densities_panda.loc[condition, selected_columns].copy()
    
    number_of_existing_atoms[step_col] = number_of_existing_atoms[step_col] * volume * 10**24        
    number_of_existing_atoms.columns = ["Isotope", "ZAMID", "Number of atoms"]

    # Extract Z, A, and excited-state flag from ZAMID
    parsed = number_of_existing_atoms["ZAMID"].apply(parse_zamid)

    number_of_existing_atoms[["Z", "A", "Excited"]] = pd.DataFrame(parsed.tolist(),
                                                                index=number_of_existing_atoms.index)
    
    # Determine the energy level for each isotope
    all_nuclide_data = pd.read_pickle(ALL_NUCLIDE_DATA)

    def determine_elevel(row):
        if row['Excited'] == 0:
            return 0.0
        else:
            matching_nuclides = all_nuclide_data[(all_nuclide_data['A'] == row['A']) & (all_nuclide_data['Z'] == row['Z'])]
            grouped = matching_nuclides.groupby(['A', 'Z', 'Par. Elevel']).size().reset_index(name='counts')
            if grouped.empty:
                return 0.0
            min_elevel = grouped['Par. Elevel'].min()
            if min_elevel > 0:
                return min_elevel
            else:
                next_highest = grouped[grouped['Par. Elevel'] > 0]['Par. Elevel'].min()
                return next_highest if pd.notna(next_highest) else 0.0

    number_of_existing_atoms['Elevel'] = number_of_existing_atoms.apply(determine_elevel, axis=1)

    # exclude the isotope if it is excited and the energy level is 0
    excluded_cases = number_of_existing_atoms[(number_of_existing_atoms['Excited'] == 1) & (number_of_existing_atoms['Elevel'] == 0)]
    excluded_cases.to_csv("excluded_isotopes.csv", index=False)
    
    number_of_existing_atoms = number_of_existing_atoms[~((number_of_existing_atoms['Excited'] == 1) & (number_of_existing_atoms['Elevel'] == 0))]

    return number_of_existing_atoms


def generate_decaypy_input(input_df, decay_time_s):
    """
    Build an NNDC decay input table from an isotopic inventory.

    Converts the inventory from number of atoms to mass (grams) using:
        mass_g = (N_atoms / N_A) * A
    where N_A is Avogadro's constant and A is treated as the molar mass in g/mol.

    The returned DataFrame contains the columns expected by the NNDC-style decay
    calculation workflow: (A, Z, Elevel, Amount (gram), decay_time (sec)).
    The same `decay_times` value is applied to every nuclide row.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Must contain at least: ["Number of atoms", "A", "Z", "Elevel"].
    decay_times : float
        Decay time in seconds applied to all nuclides.

    Returns
    -------
    pandas.DataFrame
        NNDC decay input table with columns:
        ["A", "Z", "Elevel", "Amount (gram)", "Decay_time (sec)"].
    """
    
    # Calculate the amount in grams
    amount_g = (input_df["Number of atoms"].astype(float) / Avogadro) * input_df["A"].astype(float)

    if isinstance(decay_time_s, (list, tuple)):
        decay_col = [list(decay_time_s)] * len(input_df)
    else:
        # assume scalar (float / int / numpy scalar)
        decay_col = [float(decay_time_s)] * len(input_df)

    # Create the new DataFrame
    return pd.DataFrame({
        "A": input_df["A"].astype(int),
        "Z": input_df["Z"].astype(int),
        "Elevel": input_df["Elevel"].astype(float),
        "Amount (gram)": amount_g.astype(float),
        "Decay_time (sec)": decay_col})
