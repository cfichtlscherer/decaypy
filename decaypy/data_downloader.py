import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
import tarfile
from urllib.request import urlretrieve
import urllib.request
import re


DATA_DIR = Path(__file__).parent / "data"
NUCLIDE_NAMES_FILE = DATA_DIR / "nuclide_names.txt"
ENDF_FY_DIR = DATA_DIR / "ENDF_FY"
WALLET_FILE = DATA_DIR / "nuclear_wallet_decay_data.csv"
NNDC_TXT_DIR = DATA_DIR / "nndc_txt_download"


def download_nuclide_names_pyne():
    """
    Download element symbols from the PyNE ELE.in file.
    Returns a list where index == Z (atomic number), index 0 is empty.
    """

    url = ("https://raw.githubusercontent.com/pyne/pyne/9c1d41237dc26597cd2463d5968d2ab4e2cfce0f/src/ensdf_processing/RADD/ELE.in")
    extra_elements = ["Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

    response = urllib.request.urlopen(url).read()
    text = response.decode("utf-8")
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    isotope_names = [""] + tokens + extra_elements
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(NUCLIDE_NAMES_FILE, "w", encoding="utf-8") as f:
        for symbol in isotope_names:
            f.write(symbol + "\n")
    
    print("Nuclide name download complete.")
    return 


def download_qvalues():
    """
    Q-value data cannot be downloaded automatically.

    NuDat3 generates CSV files client-side and does not expose
    direct download URLs. Please download the Q-value CSV files
    manually from https://www.nndc.bnl.gov/nudat3/:

        Qα, Qβ⁻, Qβ⁺, Qβn, QEC, QEC(p)
        → Export → CSV

    Save the files in:
        decaypy/data/Q_values/

    Expected filenames:
        nndc_Q_alpha.csv
        nndc_Q_beta_minus.csv
        nndc_Q_beta_plus.csv
        nndc_Q_beta_neutron.csv
        nndc_Q_electron_capture.csv
        nndc_Q_electron_capture_proton.csv
    """
    raise NotImplementedError(
        "Q-value data must be downloaded manually. "
        "See the function docstring for instructions."
    )


def download_endf_fission_yields():
    """
    Download and extract ENDF/B-VIII.1 fission yield files (SFY and NFY).
    The data can be downloaded from https://www.nndc.bnl.gov/endf-releases/

    - SFY: spontaneous fission yields
    - NFY: neutron-induced fission yields

    Files are downloaded as .tar.gz archives into data/ENDF_FY/,
    extracted there, and the archives are removed afterwards.
    """
    
    endf_fy_sources = {
    "sfy": "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/sfy/sfy-version.VIII.1.tar.gz",
    "nfy": "https://www.nndc.bnl.gov/endf-releases/releases/B-VIII.1/nfy/nfy-version.VIII.1.tar.gz",
    }

    for _ , url in endf_fy_sources.items():
        archive_name = url.rsplit('/', maxsplit=1)[-1]
        archive_path = ENDF_FY_DIR / archive_name

        urlretrieve(url, archive_path)

        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(ENDF_FY_DIR)

        archive_path.unlink()  # remove the .tar.gz

    print("ENDF fission yield download complete.")
    return 


def download_nuclear_decay_data_wallet_cards():
    """
    Download element symbols from the PyNE ELE.in file.
    Returns a list where index == Z (atomic number), index 0 is empty.
    """
    return


def return_isotope_name_list():
    """ this function will return a list of all isotope names.
        It contains the index shift: isotope_names[1] = "H"
    """

    data_path = Path(__file__).parent / "data" / "nuclide_names.txt"

    with open(data_path, "r", encoding="utf-8") as f:
        isotope_names = [""] + [line.strip() for line in f if line.strip()]

    return isotope_names


def mysplit(nuclide):
    """
    Split strings like '137BA' → (137, 'BA')
    """
    match = re.match(r"(\d+)([A-Za-z]+)", nuclide.strip())
    if not match:
        raise ValueError(f"Invalid nuclide format: {nuclide}")
    A, name = match.groups()
    return int(A), name.capitalize()


def return_A_and_Z_from_daughter(daughter):
    """
    daughter: e.g. '137BA'
    Returns (A, Z) = (137, 56)
    """
    A, name = mysplit(daughter)
    isotope_list = return_isotope_name_list()
    try:
        Z = isotope_list.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown element symbol in daughter: {daughter}"
        ) from exc
    return A, Z


def check_one_to_one(df, col1, col2):
    """
    Check if col1 → col2 and col2 → col1 are both one-to-one.
    """
    return df.groupby(col1)[col2].nunique().max() == 1 and \
           df.groupby(col2)[col1].nunique().max() == 1


def cleanstring(value):
    """
    Convert strings to float when possible.
    Otherwise return stripped string.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        return value

    s = value.strip()
    try:
        return float(s)
    except ValueError:
        return s


def add_stable_isotopes_from_wallet_cards(df):
    """
    Add stable isotopes from nuclear_wallet_decay_data.csv
    as decaypy rows with 'STABLE' decay mode.
    """

    column_names = ['A', 'Element', 'Z', 'N', 'Energy', 'JPi', 'Mass Exc', 'Unc', 'T1/2 (txt)', 'T1/2 (seconds)', 'Abund.', 'Unc.1', 'Dec Mode', 'Branching (%)']
    column_widths = [4, 8, 4, 4, 8, 14, 20, 10, 24, 24, 12, 8, 10, 14] # from csv file
    
    wallet_cards_df = pd.read_fwf(WALLET_FILE, widths=column_widths, names=column_names)
    
    wallet_cards_df.columns = wallet_cards_df.columns.str.replace(' ', '')
    wallet_cards_df.rename(columns={'T1/2(txt)': 'T1/2/txt'}, inplace=True)
    
    stable_isotopes_df = wallet_cards_df[wallet_cards_df['T1/2/txt'].str.strip() == 'STABLE']

    # Prepare the columns to be added
    stable_isotopes_df = stable_isotopes_df[['A', 'Element', 'Z', 'N', 'JPi']].copy()
    stable_isotopes_df['Par. Elevel'] = 0.0
    stable_isotopes_df['Unc.'] = ''
    stable_isotopes_df['Dec Mode'] = 'STABLE'
    stable_isotopes_df['T1/2/txt'] = 'STABLE'
    stable_isotopes_df['T1/2 (num)'] = 0.0
    stable_isotopes_df['Daughter'] = ''
    stable_isotopes_df['Radiation'] = ''
    stable_isotopes_df['Rad subtype'] = ''
    stable_isotopes_df['Rad Ene.'] = ''
    stable_isotopes_df['Unc'] = ''
    stable_isotopes_df['EP Ene.'] = ''
    stable_isotopes_df['Unc       .1'] = ''
    stable_isotopes_df['Rad Int.'] = ''
    stable_isotopes_df['Unc       .2'] = ''
    stable_isotopes_df['Dose'] = ''
    stable_isotopes_df['Unc       .3'] = ''
    stable_isotopes_df['Daughter_A'] = ''
    stable_isotopes_df['Daughter_Z'] = ''
    stable_isotopes_df['Element'] = stable_isotopes_df['Element'].str.upper()
    stable_isotopes_df['A'] = stable_isotopes_df['A'].astype(float)
    stable_isotopes_df['Z'] = stable_isotopes_df['Z'].astype(float)
    stable_isotopes_df['Par. Elevel'] = stable_isotopes_df['Par. Elevel'].astype(float)

    df = pd.concat([df, stable_isotopes_df], ignore_index=True)

    return df


def generate_pandas_dataframe_of_all_nuclides():
    """
    Take the data of all NNDC nuclides (downloaded as TXT files)
    and generate one large pandas DataFrame containing all of them.
    Additionally cleans and prepares the data
    """
    
    # 1) Generation of the dataframe
    footer_text = 'To save this output into a local File'
    
    all_nuclide_single_df = []

    for txt_path in sorted(NNDC_TXT_DIR.glob("*.txt")):
        nuclide_txt_content = txt_path.read_text(encoding="utf-8", errors="ignore")
        if footer_text in nuclide_txt_content:
            nuclide_txt_content = nuclide_txt_content.split(footer_text)[0].strip()
        data = nuclide_txt_content.strip()
        nuclide_df = pd.read_csv(StringIO(data), sep="\t")
        all_nuclide_single_df.append(nuclide_df)

    big_df_all_nuclide_data = pd.concat(all_nuclide_single_df)

    # remove white spaces in column names
    big_df_all_nuclide_data = big_df_all_nuclide_data.rename(columns=lambda x: x.strip())

    # 2) Cleaning of the dataframe
    cols_to_clean = ["Par. Elevel", "Daughter", "EP Ene.", "Dec Mode", "Element", "T1/2 (num)", "Radiation", "Rad Ene."]

    for col in cols_to_clean:
        big_df_all_nuclide_data[col] = big_df_all_nuclide_data[col].apply(cleanstring)
        
    big_df_all_nuclide_data["T1/2 (num)"] = big_df_all_nuclide_data["T1/2 (num)"].replace(["", " "], np.nan)
    
    big_df_all_nuclide_data.to_pickle(DATA_DIR / "data_processed" / "big_df_all_nuclide_data_NON_CLEANED.pickle")

    # 3) Making sure there is a 1-1 relationship between the halflife and the energy level
    big_df_all_nuclide_data = big_df_all_nuclide_data.drop_duplicates()

    states = big_df_all_nuclide_data[["A", "Z", "Par. Elevel", "T1/2 (num)"]].drop_duplicates()
    for z in range(4, 102):
        sel = states[states["Z"] == z]
        isotopes = sel["A"].unique()
        for a in isotopes:
            sel2 = sel[sel["A"] == a]
            if not check_one_to_one(sel2, "Par. Elevel", "T1/2 (num)"):
                energies = sel2["Par. Elevel"].unique()
                for e in energies:
                    sel3 = sel2[sel2["Par. Elevel"] == e]
                    if len(sel3) > 1:
                        sel4 = big_df_all_nuclide_data[(big_df_all_nuclide_data["Z"] == z) & (big_df_all_nuclide_data["A"] == a) & (big_df_all_nuclide_data["Par. Elevel"] == e)]
                        # find most common T/12
                        t12 = sel4.groupby("T1/2 (num)")["Par. Elevel"].count().idxmax()
                        big_df_all_nuclide_data.loc[(big_df_all_nuclide_data["Z"] == z) & (big_df_all_nuclide_data["A"] == a) & (big_df_all_nuclide_data["Par. Elevel"] == e), "T1/2 (num)"] = t12
                        print(f"Found multiple T1/2 values for same energy level, most common T1/2={t12}")
                halflives = sel2["T1/2 (num)"].unique()
                for t12 in halflives:
                    sel3 = sel2[sel2["T1/2 (num)"] == t12]
                    if len(sel3) > 1:
                        sel4 = big_df_all_nuclide_data[(big_df_all_nuclide_data["Z"] == z) & (big_df_all_nuclide_data["A"] == a) & (big_df_all_nuclide_data["T1/2 (num)"] == t12)]
                        # find most common energy level
                        elevel = sel4.groupby("Par. Elevel")["T1/2 (num)"].count().idxmax()
                        big_df_all_nuclide_data.loc[(big_df_all_nuclide_data["Z"] == z) & (big_df_all_nuclide_data["A"] == a) & (big_df_all_nuclide_data["T1/2 (num)"] == t12), "Par. Elevel"] = elevel
                        print(f"Found multiple energy values for same T1/2 level, most common E={elevel}")

    # update states / check again
    states = big_df_all_nuclide_data[["A", "Z", "Par. Elevel", "T1/2 (num)"]].drop_duplicates()

    for z in range(4, 102):
        sel = states[states["Z"] == z]
        isotopes = sel["A"].unique()
        for a in isotopes:
            sel2 = sel[sel["A"] == a]
            if not check_one_to_one(sel2, "Par. Elevel", "T1/2 (num)"):
                print(sel2)

    # 4) clean and take care of floating energy levels
    replacements = {
    "**********": "",
    "0.0+X": 1e-6,
    "0.0+Y": 2e-6,
    "0+X": 1e-6,
    "0+Y": 2e-6,
    "X": 1e-6,
    "Y": 2e-6,
    "391.7+X": 391.7 + 1e-6,
    "138.5+X": 138.5 + 1e-6,
    "49.630+X": 49.630 + 1e-6,
    "73.92+X": 73.92 + 1e-6,
    "104.0+X": 104.0 + 1e-6,
    "1347.5+X": 1347.5 + 1e-6,
    "52.4+X": 52.4 + 1e-6,
    "X+0.0": 1e-6,
    "531+X": 531.0 + 1e-6,
    "102+X": 102 + 1e-6,
    "152+X": 152 + 1e-6,
    "(100)": 100.0,
    }
    big_df_all_nuclide_data["Par. Elevel"] = big_df_all_nuclide_data["Par. Elevel"].replace(replacements)
    
    big_df_all_nuclide_data["Rad Int."] = big_df_all_nuclide_data["Rad Int."].replace({"**********": ""})
    big_df_all_nuclide_data["Rad Ene."] = big_df_all_nuclide_data["Rad Ene."].replace({"**********": ""})

    # 5) add the two columns: Daughter_A Daughter_Z
    big_df_all_nuclide_data["Daughter_A"] = big_df_all_nuclide_data["Daughter"].apply(lambda x: return_A_and_Z_from_daughter(x)[0])
    big_df_all_nuclide_data["Daughter_Z"] = big_df_all_nuclide_data["Daughter"].apply(lambda x: return_A_and_Z_from_daughter(x)[1])
    
    # 6) add stable isotopes from wallet cards
    big_df_all_nuclide_data = add_stable_isotopes_from_wallet_cards(big_df_all_nuclide_data)
    big_df_all_nuclide_data.to_pickle(DATA_DIR / "data_processed" / "big_df_all_nuclide_data.pickle")
    
    return 


def create_df_energy_level_branching():
    """
    The branching ratios between different excitation levels are not directly available
    and added here manually.
    """

    df = pd.read_pickle(DATA_DIR / "data_processed" / "big_df_all_nuclide_data.pickle")

    multiple_elevel = []

    for _, row in df.iterrows():
        all_elevel = df[(df["A"] == row["A"]) & (df["Z"] == row["Z"])]["Par. Elevel"].unique()
        all_elevel = np.sort(all_elevel)
        if np.size(all_elevel) > 2:
            multiple_elevel.append((row["A"], row["Z"]))
        elif np.size(all_elevel) == 2:
            if all_elevel[0] > 0.0:
                multiple_elevel.append((row["A"], row["Z"]))

    multiple_elevel = list(set(multiple_elevel))

    all_A_list = [i[0] for i in multiple_elevel]
    all_Z_list = [i[1] for i in multiple_elevel]

    branching_ratios_between_elevel = pd.DataFrame(columns=["A", "Z", "E1", "E2", "E3", "E4", "E2->E1", "E3->E2", "E3->E1", "E4->E3", "E4->E2", "E4->E1"])
    branching_ratios_between_elevel["A"] = all_A_list
    branching_ratios_between_elevel["Z"] = all_Z_list

    for i, A in enumerate(all_A_list):
        Z = all_Z_list[i]
        all_elevel = df[(df["A"] == A) & (df["Z"] == Z)]["Par. Elevel"].unique()
        all_elevel = np.sort(all_elevel)
        
        mask = (branching_ratios_between_elevel["A"] == A) & (branching_ratios_between_elevel["Z"] == Z)
        branching_ratios_between_elevel.loc[mask, "E1"] = all_elevel[0]
        branching_ratios_between_elevel.loc[mask, "E2"] = all_elevel[1]
        if len(all_elevel) > 2:
            branching_ratios_between_elevel.loc[mask, "E3"] = all_elevel[2]
        if len(all_elevel) > 3:
            branching_ratios_between_elevel.loc[mask, "E4"] = all_elevel[3]

    def set_branching_ratio(A, Z, column, value):
        mask = (branching_ratios_between_elevel["A"] == A) & (branching_ratios_between_elevel["Z"] == Z)
        branching_ratios_between_elevel.loc[mask, column] = value

    # All manual assignments
    set_branching_ratio(124, 51, "E3->E2", 100.0)
    set_branching_ratio(124, 51, "E3->E1", 0.0)
    set_branching_ratio(91, 41, "E3->E2", 1.09)
    set_branching_ratio(91, 41, "E3->E1", 98.01)
    set_branching_ratio(60, 25, "E3->E2", 0.0)
    set_branching_ratio(60, 25, "E3->E1", 100.0)
    set_branching_ratio(95, 45, "E3->E2", 0.0)
    set_branching_ratio(95, 45, "E3->E1", 100.0)
    set_branching_ratio(192, 77, "E3->E2", 0.0)
    set_branching_ratio(192, 77, "E3->E1", 100.0)
    set_branching_ratio(69, 32, "E3->E2", 0.0)
    set_branching_ratio(69, 32, "E3->E1", 100.0)
    set_branching_ratio(172, 71, "E3->E2", 100.0)
    set_branching_ratio(172, 71, "E3->E2", 0.0)
    set_branching_ratio(212, 85, "E3->E2", 100.0)
    set_branching_ratio(212, 85, "E3->E1", 0.0)
    set_branching_ratio(115, 52, "E3->E2", 0.0)
    set_branching_ratio(115, 52, "E3->E1", 100.0)
    set_branching_ratio(190, 77, "E3->E2", 0.0)
    set_branching_ratio(190, 77, "E3->E1", 100.0)
    set_branching_ratio(217, 89, "E4->E3", 0.0)
    set_branching_ratio(217, 89, "E4->E2", 0.0)
    set_branching_ratio(217, 89, "E4->E1", 100.0)
    set_branching_ratio(152, 63, "E3->E2", 0.0)
    set_branching_ratio(152, 63, "E3->E1", 100.0)
    set_branching_ratio(115, 51, "E3->E2", 100.0)
    set_branching_ratio(115, 51, "E3->E1", 0.0)
    set_branching_ratio(129, 49, "E3->E2", 0.0)
    set_branching_ratio(129, 49, "E3->E1", 100.0)
    set_branching_ratio(189, 82, "E2->E1", 0.0)
    set_branching_ratio(203, 82, "E2->E1", 0.0)
    set_branching_ratio(127, 50, "E3->E2", 100.0)
    set_branching_ratio(127, 50, "E3->E1", 0.0)
    set_branching_ratio(70, 29, "E3->E2", 100.0)
    set_branching_ratio(70, 29, "E3->E1", 0.0)
    set_branching_ratio(151, 69, "E3->E2", 0.0)
    set_branching_ratio(151, 69, "E3->E1", 100.0)
    set_branching_ratio(179, 72, "E2->E1", 0.0)
    set_branching_ratio(166, 67, "E3->E2", 0.0)
    set_branching_ratio(166, 67, "E3->E1", 100.0)
    set_branching_ratio(109, 48, "E3->E2", 0.0)
    set_branching_ratio(109, 48, "E3->E1", 100.0)
    set_branching_ratio(190, 74, "E3->E2", 100.0)
    set_branching_ratio(190, 74, "E3->E1", 0.0)
    set_branching_ratio(95, 47, "E3->E2", 0.0)
    set_branching_ratio(95, 47, "E3->E1", 100.0)
    set_branching_ratio(156, 65, "E3->E2", 0.0)
    set_branching_ratio(156, 65, "E3->E1", 100.0)
    set_branching_ratio(196, 79, "E3->E2", 100.0)
    set_branching_ratio(196, 79, "E3->E1", 0.0)
    set_branching_ratio(198, 81, "E3->E2", 100.0)
    set_branching_ratio(198, 81, "E3->E1", 0.0)
    set_branching_ratio(158, 65, "E3->E2", 0.0)
    set_branching_ratio(158, 65, "E3->E1", 100.0)
    set_branching_ratio(116, 47, "E3->E2", 100.0)
    set_branching_ratio(116, 47, "E3->E1", 0.0)
    set_branching_ratio(114, 49, "E3->E2", 100.0)
    set_branching_ratio(114, 49, "E3->E1", 0.0)
    set_branching_ratio(88, 39, "E3->E2", 0.0)
    set_branching_ratio(88, 39, "E3->E1", 100.0)
    set_branching_ratio(109, 49, "E3->E2", 0.0)
    set_branching_ratio(109, 49, "E3->E1", 100.0)
    set_branching_ratio(182, 73, "E3->E2", 0.0)
    set_branching_ratio(182, 73, "E3->E1", 100.0)

    branching_ratios_between_elevel.to_pickle(DATA_DIR / "data_processed" / "branching_ratios_between_Elevel.pickle")

    return
