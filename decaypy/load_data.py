from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Dict, Tuple

import pandas as pd

from decaypy.data_downloader import return_isotope_name_list


@dataclass(frozen=True)
class DecayData:
    """
    Container for all nuclear datasets used by decaypy.

    This object loads and cleans the data once. Other parts of the code
    (decay graph building, matrix building, emissions) should treat these
    tables and lookup dicts as read-only.

    Attributes
    ----------
    ensdf : pd.DataFrame
        Main ENSDF/NNDC-derived nuclide dataset (your big_df...pickle), with
        normalized dtypes where practical.
    branching_between_elevels : pd.DataFrame
        Precomputed branching ratios between excited levels.
    wallet_cards : pd.DataFrame
        Nuclear Wallet Cards extracted table, cleaned and normalized.
    isotope_name_list : list[str]
        List mapping Z -> element symbol (as you currently use it).
    q_beta_minus, q_beta_plus, q_alpha, q_ec : dict
        Q-value lookup dicts keyed by (n, z) -> keV as float.
    """

    ensdf: pd.DataFrame
    branching_between_elevels: pd.DataFrame
    wallet_cards: pd.DataFrame
    isotope_name_list: list

    q_beta_minus: Dict[Tuple[int, int], float]
    q_beta_plus: Dict[Tuple[int, int], float]
    q_alpha: Dict[Tuple[int, int], float]
    q_ec: Dict[Tuple[int, int], float]


def load_decay_data(*, include_endf_decay_data: bool = False) -> DecayData:
    """
    Load all nuclear datasets used by decaypy and return a DecayData container.

    Parameters
    ----------
    include_endf_decay_data : bool, optional
        If True, load 'big_df_all_nuclide_data_including_ONIX_ENDF.pickle'.
        Otherwise load 'big_df_all_nuclide_data.pickle'.

    Returns
    -------
    DecayData
        A frozen dataclass containing cleaned DataFrames and lookup dicts.
    """
    # ---- Load pickles ----
    branching_between_elevels = pd.read_pickle(
        files("decaypy.data.data_processed").joinpath("branching_ratios_between_Elevel.pickle")
    )

    if include_endf_decay_data:
        ensdf = pd.read_pickle(
            files("decaypy.data.data_processed").joinpath("big_df_all_nuclide_data_including_ONIX_ENDF.pickle")
        )
    else:
        ensdf = pd.read_pickle(
            files("decaypy.data.data_processed").joinpath("big_df_all_nuclide_data.pickle")
        )

    # ---- Load Q-value tables and build lookup dicts ----
    q_beta_minus_df = pd.read_csv(files("decaypy.data.Q_values").joinpath("nndc_Q_beta_minus.csv"))
    q_beta_plus_df = pd.read_csv(files("decaypy.data.Q_values").joinpath("nndc_Q_beta_plus.csv"))
    q_alpha_df = pd.read_csv(files("decaypy.data.Q_values").joinpath("nndc_Q_alpha.csv"))
    q_ec_df = pd.read_csv(files("decaypy.data.Q_values").joinpath("nndc_Q_electron_capture.csv"))

    q_beta_minus = {(int(r["n"]), int(r["z"])): float(r["betaMinus(keV)"]) for _, r in q_beta_minus_df.iterrows()}
    q_beta_plus = {(int(r["n"]), int(r["z"])): float(r["positronEmission(keV)"]) for _, r in q_beta_plus_df.iterrows()}
    q_alpha = {(int(r["n"]), int(r["z"])): float(r["alpha(keV)"]) for _, r in q_alpha_df.iterrows()}
    q_ec = {(int(r["n"]), int(r["z"])): float(r["electronCapture(keV)"]) for _, r in q_ec_df.iterrows()}

    # ---- Wallet Cards ----
    wallet_cards = pd.read_csv(
        files("decaypy.data").joinpath("nuclear_wallet_decay_data.csv"),
        sep="\t",
    )
    wallet_cards = _clean_wallet_cards(wallet_cards)

    # ---- Element symbol list ----
    isotope_name_list = return_isotope_name_list()

    # ---- Normalize ENSDF dtypes (important: do this ONCE here) ----
    ensdf = _normalize_ensdf_types(ensdf)

    return DecayData(
        ensdf=ensdf,
        branching_between_elevels=branching_between_elevels,
        wallet_cards=wallet_cards,
        isotope_name_list=isotope_name_list,
        q_beta_minus=q_beta_minus,
        q_beta_plus=q_beta_plus,
        q_alpha=q_alpha,
        q_ec=q_ec,
    )


def _normalize_ensdf_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dtypes for frequently used ENSDF columns.

    We avoid mutating caller-owned frames by returning a copy.
    """
    df = df.copy()

    # Columns you frequently compare numerically
    for col in ["A", "Z", "Par. Elevel", "T1/2 (num)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Radiation intensity is often messy; keep numeric where possible
    if "Rad Int." in df.columns:
        df["Rad Int."] = pd.to_numeric(df["Rad Int."], errors="coerce")

    # Radiation energy can be numeric but sometimes non-numeric; keep as-is unless you want to coerce
    # If you want numeric-only downstream, do coercion locally in the emissions layer.

    return df


def _clean_wallet_cards(nwc: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the Nuclear Wallet Cards extracted table.

    - Strips column names
    - Converts A, Z, Energy, half-life, branching into floats where possible
    - Cleans odd branching string markers (<, >, ?, etc.)
    """
    nwc = nwc.copy()
    nwc.rename(columns=lambda x: x.strip(), inplace=True)

    # Apply cleanstring-like behavior
    for col in ["A", "Z", "Energy", "Dec Mode", "T1/2 (seconds)", "Branching (%)"]:
        if col in nwc.columns:
            nwc[col] = nwc[col].apply(_cleanstring)

    if "Branching (%)" in nwc.columns:
        nwc["Branching (%)"] = nwc["Branching (%)"].apply(_clean_branching_wallet_cards)
        nwc["Branching (%)"] = pd.to_numeric(nwc["Branching (%)"], errors="coerce").fillna(0.0)

    # Make numeric columns numeric if possible
    for col in ["A", "Z", "Energy", "T1/2 (seconds)"]:
        if col in nwc.columns:
            nwc[col] = pd.to_numeric(nwc[col], errors="coerce")

    return nwc


def _cleanstring(s):
    """Convert strings that start with a float to float; keep other entries as-is."""
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return 0.0
        parts = s.split()
        try:
            return float(parts[0])
        except (ValueError, TypeError):
            return s
    return s


def _clean_branching_wallet_cards(val):
    """Strip non-numeric markers commonly found in wallet cards branching fields."""
    if isinstance(val, float) or isinstance(val, int):
        return val
    if not isinstance(val, str):
        return val

    s = val
    for ch in ["<", ">", "?", "=", "@", "~"]:
        s = s.replace(ch, " ")
    s = s.replace(" ", "").strip()
    if s == "":
        s = "0.0"
    return s

