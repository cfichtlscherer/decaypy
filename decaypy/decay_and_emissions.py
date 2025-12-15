import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.constants import Avogadro

import openmc
import openmc.deplete

from decaypy.load_data import load_decay_data
from decaypy.determine_emissions import GammaEmissions
from decaypy.determine_decay_intensities import DecayIntensities
from decaypy.decay_chain_solver import DecayChainSolver
from decaypy.nuclide_data_helper import NuclideDataHelper

from importlib.resources import files

class Decay:
    def __init__(self,
                 include_sf=True,
                 fission_yield_reduction=False,
                 induced_fission_rates=None,
                 include_ENDF_decay_data=False,
                 verbosity=False,
                 data=None):
        
        self.include_sf = include_sf
        self.fission_yield_reduction = fission_yield_reduction
        self.induced_fission_rates = induced_fission_rates or {}
        self.verbosity = verbosity

        # load the data
        self.data = data or load_decay_data(include_endf_decay_data=include_ENDF_decay_data)

        # initialize specialized modules
        self.helper = NuclideDataHelper(self.data, self)
        self.intensities = DecayIntensities(self.data, self, self.helper)
        self.emissions = GammaEmissions(self.data, self)
        self.chain_solver = DecayChainSolver(self.data, self)

        self.decays_cache = {}
        

    def get_fission_children(self, A, Z, Elevel, fission_type):
        """ this method creates a df like the one in get_all_decays with fission that
            can be combined with the other decays"""

        FY_list = self.helper.return_fission_yields(A, Z, Elevel, fission_type=fission_type)

        required_columns = ["A_parent", "Element_parent", "Z_parent", "A_daughter",
                            "Elevel_daughter", "Z_daughter", "Dec Mode", "T1/2 (num)",
                            "Rad Ene.", "Rad Int."]
        if not FY_list:
            return pd.DataFrame(columns=required_columns)

        element_parent = self.data.isotope_name_list[int(Z)].upper()
        hl_parent = self.helper.get_from_nuclide_Elevel_half_life(A, Z, Elevel)

        decay_records = []
        for sf_product in FY_list:

            daughter_info, rad_intensity = sf_product
            A_daughter, Z_daughter, Elevel_daughter = daughter_info

            # Construct the decay record as a dictionary
            decay_record = {
                "A_parent": A,
                "Element_parent": element_parent,
                "Z_parent": Z,
                "A_daughter": A_daughter,
                "Elevel_daughter": Elevel_daughter,
                "Z_daughter": Z_daughter,
                "Dec Mode": fission_type,
                "T1/2 (num)": hl_parent,
                "Rad Ene.": np.nan,
                "Rad Int.": rad_intensity
            }

            decay_records.append(decay_record)

        nuclide_all_sf_decay_df = pd.DataFrame(decay_records, columns=required_columns)
        return nuclide_all_sf_decay_df


    def get_all_decays(self, A, Z, Elevel):
        """ this function returns for a given nuclide all relevant decays:
        There are A, BM, BP, EC decays in which the nuclide becomes another.
        G decays are only considered if the parent is the same as the daughter
        The Elevel can be:
            a) a float to get the decays from a specific Elevel
            b) a tuple [lower_bound, upper_bound] to get all decays that start in this bound
            c) the string "all" to get all decays of this nuclide
        """

        cache_key = (A, Z, Elevel)
        if cache_key in self.decays_cache:
            return self.decays_cache[cache_key]

        # 1) checking if all energy levels are requested, only the ones in a certain area, or a discrete level
        if Elevel == "all":
            nuclide_df = self.data.ensdf[(self.data.ensdf["A"] == A) & (self.data.ensdf["Z"] == Z)]
        elif isinstance(Elevel, tuple):
            nuclide_df = self.data.ensdf[(self.data.ensdf["A"] == A) & (self.data.ensdf["Z"] == Z) & (self.data.ensdf["Par. Elevel"] >= Elevel[0]) & (self.data.ensdf["Par. Elevel"] <= Elevel[1])]
        else:
            nuclide_df = self.data.ensdf[(self.data.ensdf["A"] == A) & (self.data.ensdf["Z"] == Z) & (self.data.ensdf["Par. Elevel"] == Elevel)]

        nuclide_df.reset_index()
        all_decays = []

        # determine the fraction of fissions in the decay
        hl_parent = self.helper.get_from_nuclide_Elevel_half_life(A, Z, Elevel)

        lambda_natural = np.log(2) / hl_parent if hl_parent > 0 else 0.0

        lambda_induced, fission_type = self.induced_fission_rates.get((A, Z, Elevel), (0.0, None))

        lambda_total = lambda_natural + lambda_induced
        half_life_updated = np.log(2) / lambda_total if lambda_total > 0 else 0.0


        decay_branching_scaling_factor = lambda_natural / lambda_total if lambda_total > 0 else 1.0

        decay_records = []

        # 2) going through all decays with the chosen A, Z, Elevel
        for _, row in nuclide_df.iterrows():
            try:
                rad_int = float(row["Rad Int."]) * decay_branching_scaling_factor
            except ValueError:
                print(f"Cannot convert '{row['Rad Int.']}' to float.")
                rad_int = 0

            decay_record = {"A_parent": A,
                            "Z_parent": Z,
                            "Element_parent": row["Element"],
                            "Element_daughter": row["Daughter"].lstrip('0123456789'),
                            "Rad Ene.": row["Rad Ene."],
                            "Rad Int.": rad_int}

            if (row["Dec Mode"] == "B-") & (row["Radiation"] == "BM"):
                q_value = self.data.q_beta_minus[(int(A - Z), int(Z))]
                decay_record["Z_daughter"] = Z + 1
                decay_record["A_daughter"] = A
                decay_record["Dec Mode"] = "BM"

            elif row["Radiation"] == "BP":
                q_value = self.data.q_beta_plus[(int(A - Z), int(Z))]
                decay_record["Z_daughter"] = Z - 1
                decay_record["A_daughter"] = A
                decay_record["Dec Mode"] = "BP"

            elif (row["Dec Mode"] == "EC"):
                q_value = self.data.q_ec[(int(A - Z), int(Z))]
                decay_record["Z_daughter"] = Z - 1
                decay_record["A_daughter"] = A
                decay_record["Dec Mode"] = "EC"

            elif row["Radiation"] == "A":
                q_value = self.data.q_alpha[(int(A - Z), int(Z))]
                decay_record["Z_daughter"] = Z - 2
                decay_record["A_daughter"] = A - 4
                decay_record["Dec Mode"] = "A"

            elif row["Dec Mode"] == "IT":
                decay_record["Z_daughter"] = Z
                decay_record["A_daughter"] = A
                decay_record["Dec Mode"] = "IT"

            else:
                continue

            if row["Radiation"] == "G":
                decay_record["Elevel_daughter"] = row["Par. Elevel"] - float(row["Rad Ene."])
            elif row["EP Ene."] in ["", None]:
                decay_record["Elevel_daughter"] = 0.0
            else:
                # EP Energy is the endpoint energy for a specific decay branch: 
                # EP = Qbeta − daughter excitation energy.
                decay_record["Elevel_daughter"] = q_value - float(row["EP Ene."])

            decay_records.append(decay_record)

        if decay_records:
            decay_df = pd.DataFrame(decay_records, columns=["A_parent", "Element_parent", "Z_parent",
                                                            "A_daughter", "Element_daughter", "Z_daughter",
                                                            "Elevel_daughter", "Dec Mode", "T1/2 (num)",
                                                            "Rad Ene.", "Rad Int."])
            all_decays.append(decay_df)

        # 5) add the induced fission to decay_df
        if lambda_induced > 0.0:
            fission_children_df = self.get_fission_children(A, Z, Elevel, fission_type=fission_type).dropna(how='all')
            fission_children_df["Rad Int."] = fission_children_df["Rad Int."] * (lambda_induced / lambda_total) *100         # as the intensity is given in percent
            fission_children_df["Element_daughter"] = fission_children_df["Z_daughter"].apply(lambda z: self.data.isotope_name_list[int(z)].upper())
            all_decays.append(fission_children_df)

        # 4) add the SF to the decay_df
        if self.include_sf:
            # if no SF data is there a NaN line is returned which is removed here.
            spontaneous_fission_children_df = self.get_fission_children(A, Z, Elevel, fission_type="SF").dropna(how='all')
            spontaneous_fission_children_df["Element_daughter"] = spontaneous_fission_children_df["Z_daughter"].apply(lambda z: self.data.isotope_name_list[int(z)].upper())
            spontaneous_fission_children_df["Rad Int."] *= decay_branching_scaling_factor *100      # as the intensity is given in percent
            all_decays.append(spontaneous_fission_children_df)

        # Create and return decay_df
        if not all_decays:
            nuclide_all_decay_df = pd.DataFrame(columns=["A_parent", "Element_parent", "Z_parent", "A_daughter", "Element_daughter", "Z_daughter",
                                                        "Elevel_daughter", "Dec Mode", "T1/2 (num)", "Rad Ene.", "Rad Int."])
            return nuclide_all_decay_df
        
        # to remove empty dataframes from the list - creates a pd Warning
        valid_decays = [df for df in all_decays if df is not None and not df.empty and not df.isna().all().all()]
        if valid_decays:
            nuclide_all_decay_df = pd.concat(valid_decays, ignore_index=True)
        else:
            nuclide_all_decay_df = pd.DataFrame()
        nuclide_all_decay_df["T1/2 (num)"] = half_life_updated

        self.decays_cache[cache_key] = nuclide_all_decay_df

        return nuclide_all_decay_df


    def get_children_one_generation(self, A, Z, Elevel):
        """ input:  nuclide and its Elevel
            output: a list of all children
            children are 5-tuples of (A_child, Z_child, A_parent, Z_parent, decay mode)
            example: (137.0, 55.0, 137.0, 54.0, 'BM')"""

        decay_df = self.get_all_decays(A, Z, Elevel)
        all_children = []
        for _, row in decay_df.iterrows():
            if row["Dec Mode"] != "IT":
                all_children.append((row["A_daughter"], row["Z_daughter"], A, Z, row["Dec Mode"], row['T1/2 (num)'], row['Rad Int.']))

        all_children = list(set(all_children))

        return all_children


    def return_all_reached_elevel(self, A, Z, Elevel, child):
        """
        Returns all reached energy levels for a given child nuclide.

        Args:
            A (int): Atomic mass number of the parent nuclide.
            Z (int): Atomic number of the parent nuclide.
            Elevel (float or tuple): Energy level of the parent nuclide.
            child (tuple): A tuple containing the atomic mass number, atomic number, decay mode, and other information of the child nuclide.

        Returns:
        list: A sorted list of all reached energy levels for the child nuclide.
        """
        q_value = self.helper.return_q_value(A, Z, child)

        different_Elevel_of_child = self.data.ensdf[(self.data.ensdf["A"] == child[0]) & (self.data.ensdf["Z"] == child[1])]["Par. Elevel"].unique()
        nuclide_all_decay_df = self.get_all_decays(A, Z, Elevel).copy()
        nuclide_all_decay_df = nuclide_all_decay_df[nuclide_all_decay_df["Dec Mode"] == child[4]]

        # key mod
        if child[4] in ['SF', 'FF', 'DTF', 'TF']:
            nuclide_all_decay_df = nuclide_all_decay_df[(nuclide_all_decay_df["A_daughter"] == child[0]) & (nuclide_all_decay_df["Z_daughter"] == child[1])]
        # end key mod

        non_numeric_values = nuclide_all_decay_df[pd.to_numeric(nuclide_all_decay_df["Rad Ene."], errors='coerce').isnull()]
        if (not non_numeric_values.empty) and self.verbosity and (child[4] not in ['SF', 'FF', 'DTF', 'TF']):
            print("Non-numeric values found in 'Rad Ene.' (decay is dropped):")
            print(non_numeric_values)

        if child[4] not in ['SF', 'FF', 'DTF', 'TF']:
            # we need to kick them out for getting decay emissions but need them in for decay with fission
            nuclide_all_decay_df = nuclide_all_decay_df[pd.to_numeric(nuclide_all_decay_df["Rad Ene."], errors='coerce').notnull()]

        nuclide_all_decay_df.loc[:, "Energy Difference"] = q_value - nuclide_all_decay_df["Rad Ene."]
        nuclide_all_decay_df.loc[:, "closest_Elevel_child"] = nuclide_all_decay_df["Elevel_daughter"].apply(lambda x: self.helper.get_closest_or_next_lowest(different_Elevel_of_child, x))
        all_reached_elevel_of_child = nuclide_all_decay_df["closest_Elevel_child"].unique()
        all_reached_elevel_of_child = np.sort(all_reached_elevel_of_child)

        intensities_reached_elevel = np.asarray(
            [np.sum(nuclide_all_decay_df[nuclide_all_decay_df["closest_Elevel_child"] == reached_Elevel]["Rad Int."]) for reached_Elevel in all_reached_elevel_of_child])

        if len(all_reached_elevel_of_child) == 0:
            # if no reached Elevel is found, I assume the child decays to the ground level with an intensity of 100%
            all_reached_elevel_of_child = [0.0]
            intensities_reached_elevel = [100.0]

        return all_reached_elevel_of_child, intensities_reached_elevel

    

    def fill_nc_basic_information(self, A, Z, Elevel, generation, nuclide_chain, reached_Elevel, intensity, child, type_of_decay):

        nuclide_chain["A_parent"] = A
        nuclide_chain["Z_parent"] = Z
        nuclide_chain["Element parent"] = self.data.ensdf[self.data.ensdf["Z"] == Z]["Element"].unique()[0]
        nuclide_chain["Elevel"] = Elevel
        nuclide_chain["T1/2 (num)"] = self.helper.get_from_nuclide_Elevel_half_life(A, Z, Elevel)
        nuclide_chain["Generation"] = generation
        nuclide_chain["Total A,BM,BP,EC Intensity"] = self.intensities.return_total_decay_intensity(A, Z, Elevel) + self.intensities.return_EC_decay_intensity(A, Z, Elevel)

        if type_of_decay == "deexcitation":
            nuclide_chain["A_child"] = A
            nuclide_chain["Z_child"] = Z
            try:
                nuclide_chain["Element child"] = self.data.ensdf[self.data.ensdf["Z"] == Z]["Element"].unique()[0]
            except KeyError:
                nuclide_chain["Element child"] = "Unknown"
            nuclide_chain["Dec Mode"] = "dex"
            nuclide_chain["Elevel_child"] = reached_Elevel
            if float(nuclide_chain["Total A,BM,BP,EC Intensity"]) > 100:
                nuclide_chain["Intensity"] = 0
            else:
                nuclide_chain["Intensity"] = (100 - float(nuclide_chain["Total A,BM,BP,EC Intensity"])) * intensity / 100


            if len(self.data.ensdf[(self.data.ensdf["Z"] == Z) & (self.data.ensdf["A"] == A) & (self.data.ensdf["Par. Elevel"] == reached_Elevel)]) > 0:
                nuclide_chain["T1/2 (num) child"] = self.helper.get_from_nuclide_Elevel_half_life(A, Z, reached_Elevel)
            else:
                # no data of child in ENSDF database -> stable
                nuclide_chain["T1/2 (num) child"] = 0.0

        elif type_of_decay == "EC":
            nuclide_chain["A_child"] = A
            nuclide_chain["Z_child"] = Z-1
            # some isotopes do not exist in the df at all
            nuclide_chain["Element child"] = self.data.isotope_name_list[int(Z-1)].upper()
            nuclide_chain["Dec Mode"] = "EC"
            nuclide_chain["Elevel_child"] = 0.0
            nuclide_chain["Intensity"] = intensity
            if len(self.data.ensdf[(self.data.ensdf["Z"] == Z-1) & (self.data.ensdf["A"] == A) & (self.data.ensdf["Par. Elevel"] == 0.0)]) > 0:
                nuclide_chain["T1/2 (num) child"] = self.helper.get_from_nuclide_Elevel_half_life(A, Z-1, 0.0)
            else:
                # no data of child in ENSDF database -> stable
                nuclide_chain["T1/2 (num) child"] = 0.0

        else:
            nuclide_chain["A_child"] = child[0]
            nuclide_chain["Z_child"] = child[1]
            try:
                nuclide_chain["Element child"] = self.data.ensdf[self.data.ensdf["Z"] == child[1]]["Element"].unique()[0]
            except KeyError:
                nuclide_chain["Element child"] = "Unknown"

            nuclide_chain["Dec Mode"] = child[4]

            if nuclide_chain["Dec Mode"] == "A":
                nuclide_chain["Q value"] = self.data.q_alpha[(nuclide_chain["A_parent"] - nuclide_chain["Z_parent"], nuclide_chain["Z_parent"])]
            elif nuclide_chain["Dec Mode"] == "BM":
                nuclide_chain["Q value"] = self.data.q_beta_minus[(nuclide_chain["A_parent"] - nuclide_chain["Z_parent"], nuclide_chain["Z_parent"])]
            elif nuclide_chain["Dec Mode"] == "BP":
                nuclide_chain["Q value"] = self.data.q_beta_plus[(nuclide_chain["A_parent"] - nuclide_chain["Z_parent"], nuclide_chain["Z_parent"])]
            elif nuclide_chain["Dec Mode"] == "EC":
                nuclide_chain["Q value"] = self.data.q_ec[(nuclide_chain["A_parent"] - nuclide_chain["Z_parent"], nuclide_chain["Z_parent"])]
            elif nuclide_chain["Dec Mode"] == "no data on child":
                nuclide_chain['Elevel_child'] = ""
                nuclide_chain['T1/2 (num) child'] = ""

            if type_of_decay == "decay_to_a_child":
                nuclide_chain["Elevel_child"] = reached_Elevel
                nuclide_chain["T1/2 (num) child"] = self.helper.get_from_nuclide_Elevel_half_life(child[0], child[1], reached_Elevel)
                nuclide_chain["Intensity"] = intensity

        return nuclide_chain


    def decay_single_isotope(self, A, Z, Elevel, amount_gram, time_seconds):
        """Wrapper. Delegates to chain_solver module."""
        return self.chain_solver.decay_single_isotope(A, Z, Elevel, amount_gram, time_seconds)

    def decay_isotopic_mixture(self, isotopic_mixture):
        """Wrapper. Delegates to chain_solver module."""
        return self.chain_solver.decay_isotopic_mixture(isotopic_mixture)
    
    def return_isotope_gamma_emissions(self, A, Z, Elevel):
        """Wrapper for backward compatibility. Delegates to emissions module."""
        return self.emissions.return_isotope_gamma_emissions(A, Z, Elevel)
    
    def return_gamma_emissions_isotopic_mixture(self, isotopic_mixture, add_iso_cols=False):
        """Wrapper for backward compatibility. Delegates to emissions module."""
        return self.emissions.return_gamma_emissions_isotopic_mixture(isotopic_mixture, add_iso_cols=add_iso_cols)
