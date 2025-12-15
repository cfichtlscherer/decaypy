import numpy as np
import pandas as pd
import scipy.sparse as sp
import openmc.deplete.cram
from scipy.constants import Avogadro


class DecayChainSolver:
    """
    Handles decay chain building and CRAM48 matrix solving.
    """
    
    def __init__(self, data, decay_instance):
        """
        Parameters
        ----------
        data : DecayData
            The loaded decay data object
        decay_instance : Decay
            Reference to parent Decay class instance
        """
        self.data = data
        self.decay = decay_instance
        self.intensities = decay_instance.intensities
        self.helper = decay_instance.helper

        # Cache for decay matrices
        self.decay_matrices_cache = {}

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
        if cache_key in self.decay.decays_cache:
            return self.decay.decays_cache[cache_key]

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

        lambda_induced, fission_type = self.decay.induced_fission_rates.get((A, Z, Elevel), (0.0, None))

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
            fission_children_df = self.decay.get_fission_children(A, Z, Elevel, fission_type=fission_type).dropna(how='all')
            fission_children_df["Rad Int."] = fission_children_df["Rad Int."] * (lambda_induced / lambda_total) *100         # as the intensity is given in percent
            fission_children_df["Element_daughter"] = fission_children_df["Z_daughter"].apply(lambda z: self.data.isotope_name_list[int(z)].upper())
            all_decays.append(fission_children_df)

        # 4) add the SF to the decay_df
        if self.decay.include_sf:
            # if no SF data is there a NaN line is returned which is removed here.
            spontaneous_fission_children_df = self.decay.get_fission_children(A, Z, Elevel, fission_type="SF").dropna(how='all')
            spontaneous_fission_children_df["Element_daughter"] = spontaneous_fission_children_df["Z_daughter"].apply(lambda z: self.data.isotope_name_list[int(z)].upper())
            spontaneous_fission_children_df["Rad Int."] *= decay_branching_scaling_factor *100      # as the intensity is given in percent
            all_decays.append(spontaneous_fission_children_df)

        # Create and return decay_df
        if all_decays == []:
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

        self.decay.decays_cache[cache_key] = nuclide_all_decay_df

        return nuclide_all_decay_df

    
    def get_total_nuclide_chain(self, A, Z, Elevel):
        """Build complete decay chain for a nuclide.
        
        Returns DataFrame with all decay pathways including deexcitations,
        EC decays, and transmutations to daughter nuclei.
        
        Parameters
        ----------
        A : int
            Mass number
        Z : int
            Atomic number
        Elevel : float
            Energy level
            
        Returns
        -------
        pd.DataFrame
            Complete nuclide chain
        """
        index, generation = 0, 0
        all_lines = []
        open_nuclides = [(A, Z, Elevel)]
        nuclide_chain = {"index": index}

        while len(open_nuclides) > 0:
            if self.decay.verbosity:
                print("Decay chains to calculate: ", len(open_nuclides))
                
            A, Z, Elevel = open_nuclides.pop(0)
            
            if Elevel > 0.0:
                all_elevel = (
                    self.helper.return_all_half_lifes_or_Elevel_for_a_nuclide(
                        A, Z, return_hl=False
                    )
                )
                index_elevel = np.where(all_elevel == Elevel)[0][0]

                if (index_elevel > 1) or (
                    (index_elevel == 1) & (all_elevel[0] > 0.0)
                ):
                    l = self.data.branching_between_elevels[
                        (self.data.branching_between_elevels["A"] == A) & 
                        (self.data.branching_between_elevels["Z"] == Z)
                    ].reset_index(drop=True)
                    
                    for transition in [
                        "E2->E1", "E3->E2", "E3->E1", "E4->E3", "E4->E2", "E4->E1"
                    ]:
                        transition_intensity = l._get_value(0, transition)
                        if transition_intensity > 0:
                            elevel_result = all_elevel[int(transition[-1]) - 1]
                            nuclide_chain = {"index": index}
                            nuclide_chain = (
                                self.decay.fill_nc_basic_information(
                                    A, Z, Elevel, generation, nuclide_chain, 
                                    reached_Elevel=elevel_result, 
                                    intensity=transition_intensity, 
                                    child="", type_of_decay="deexcitation"
                                )
                            )
                            all_lines.append(nuclide_chain)
                            if len(
                                self.data.ensdf[
                                    (self.data.ensdf["Z"] == Z) & 
                                    (self.data.ensdf["A"] == A) & 
                                    (self.data.ensdf["Par. Elevel"] == elevel_result)
                                ]
                            ) > 0:
                                open_nuclides.append((A, Z, elevel_result))
                            index += 1

                else:
                    nuclide_chain = {"index": index}
                    nuclide_chain = (
                        self.decay.fill_nc_basic_information(
                            A, Z, Elevel, generation, nuclide_chain, 
                            reached_Elevel=0.0, intensity=100.0, 
                            child="", type_of_decay="deexcitation"
                        )
                    )
                    all_lines.append(nuclide_chain)
                    if len(
                        self.data.ensdf[
                            (self.data.ensdf["Z"] == Z) & 
                            (self.data.ensdf["A"] == A) & 
                            (self.data.ensdf["Par. Elevel"] == 0.0)
                        ]
                    ) > 0:
                        open_nuclides.append((A, Z, 0.0))
                    index += 1

            intensity = self.intensities.return_EC_decay_intensity(
                A, Z, Elevel
            )
            if intensity > 0:
                nuclide_chain = {"index": index}
                nuclide_chain = (
                    self.decay.fill_nc_basic_information(
                        A, Z, Elevel, generation, nuclide_chain,
                        reached_Elevel=0.0, intensity=intensity,
                        child="", type_of_decay="EC"
                    )
                )
                all_lines.append(nuclide_chain)
                if len(
                    self.data.ensdf[
                        (self.data.ensdf["Z"] == Z-1) & 
                        (self.data.ensdf["A"] == A) & 
                        (self.data.ensdf["Par. Elevel"] == 0.0)
                    ]
                ) > 0:
                    open_nuclides.append((A, Z-1, 0.0))
                index += 1

            all_children = self.decay.get_children_one_generation(A, Z, Elevel)

            for child in all_children:
                if len(
                    self.data.ensdf[
                        (self.data.ensdf["A"] == child[0]) & 
                        (self.data.ensdf["Z"] == child[1])
                    ]
                ) == 0:
                    nuclide_chain = {"index": index}
                    nuclide_chain = (
                        self.decay.fill_nc_basic_information(
                            A, Z, Elevel, generation, nuclide_chain, 
                            reached_Elevel="", intensity="", 
                            child=child, type_of_decay="no_child_data"
                        )
                    )
                    all_lines.append(nuclide_chain)
                    index += 1

                else:
                    all_reached_elevel_of_child, intensities_reached_elevel = (
                        self.decay.return_all_reached_elevel(A, Z, Elevel, child)
                    )
                    for re_index, reached_Elevel in enumerate(
                        all_reached_elevel_of_child
                    ):
                        nuclide_chain = {"index": index}
                        nuclide_chain = (
                            self.decay.fill_nc_basic_information(
                                A, Z, Elevel, generation, nuclide_chain, 
                                reached_Elevel, 
                                intensity=intensities_reached_elevel[re_index], 
                                child=child, type_of_decay="decay_to_a_child"
                            )
                        )
                        all_lines.append(nuclide_chain)
                        open_nuclides.append((child[0], child[1], reached_Elevel))
                        index += 1

            open_nuclides = list(set(open_nuclides))

            # Fill empty line if there is no decay
            if len(all_lines) == 0:
                child = (A, Z, A, Z, "no data on child")
                nuclide_chain = {"index": index}
                nuclide_chain = (
                    self.decay.fill_nc_basic_information(
                        A, Z, Elevel, generation, nuclide_chain, 
                        reached_Elevel="", intensity="", 
                        child=child, type_of_decay="no_child_data"
                    )
                )
                all_lines.append(nuclide_chain)
                index += 1

            generation += 1
            nuclide_chain = pd.DataFrame(all_lines)

        nuclide_chain = nuclide_chain.drop_duplicates(
            ["A_parent", "Z_parent", "Elevel", "A_child", "Z_child", "Elevel_child", "Dec Mode"]
        ).reset_index(drop=True)
        
        return nuclide_chain

    
    def generate_decay_matrix(self, nuclide_chain):
        """Generate sparse decay matrix from nuclide chain.
        
        Parameters
        ----------
        nuclide_chain : pd.DataFrame
            Decay chain from get_total_nuclide_chain()
            
        Returns
        -------
        tuple
            (decay_matrix as csr_matrix, all_isotopes_in_chain DataFrame)
        """
        if nuclide_chain.empty:
            raise ValueError("The nuclide chain is empty.")

        nuclide_chain.fillna(0, inplace=True)

        # Update half-life for induced fission losses
        for index, row in nuclide_chain.iterrows():
            if (row["A_parent"], row["Z_parent"], row["Elevel"]) in (
                self.decay.induced_fission_rates.keys()
            ):
                new_lambda = (
                    (np.log(2) / row["T1/2 (num)"]) + 
                    self.decay.induced_fission_rates[
                        (row["A_parent"], row["Z_parent"], row["Elevel"])
                    ][0]
                )
                new_half_life = np.log(2) / new_lambda
                nuclide_chain.at[index, "T1/2 (num)"] = new_half_life

        # Fill missing columns
        if 'Elevel_child' not in nuclide_chain.columns:
            nuclide_chain['Elevel_child'] = ''
        if 'T1/2 (num) child' not in nuclide_chain.columns:
            nuclide_chain['T1/2 (num) child'] = ''

        all_isotopes_in_chain = pd.concat([
            nuclide_chain[[
                'A_parent', 'Z_parent', 'Element parent', 'Elevel', 'T1/2 (num)'
            ]].rename(columns={
                'A_parent': 'A', 'Z_parent': 'Z', 'Element parent': 'Element'
            }),
            nuclide_chain[[
                'A_child', 'Z_child', 'Element child', 'Elevel_child', 'T1/2 (num) child'
            ]].rename(columns={
                'A_child': 'A', 'Z_child': 'Z', 'Element child': 'Element',
                'T1/2 (num) child': "T1/2 (num)", "Elevel_child": "Elevel"
            })
        ], axis=0)

        all_isotopes_in_chain = all_isotopes_in_chain.drop_duplicates().reset_index(
            drop=True
        )
        all_isotopes_in_chain = all_isotopes_in_chain[
            all_isotopes_in_chain['T1/2 (num)'] != ''
        ]

        decay_matrix = np.zeros(
            (len(all_isotopes_in_chain), len(all_isotopes_in_chain))
        )

        for index, row in all_isotopes_in_chain.iterrows():
            if row["T1/2 (num)"] == 0.0:
                # stable isotopes
                decay_matrix[index, index] = 0.0
            else:
                decay_matrix[index, index] = - np.log(2) / row["T1/2 (num)"]

        for index, row in nuclide_chain.iterrows():
            if (
                (row["T1/2 (num)"] != "-") & 
                (row["T1/2 (num)"] != 0.0) & 
                (~np.isnan(row["T1/2 (num)"]))
            ):
                parent_index = all_isotopes_in_chain[
                    (all_isotopes_in_chain["A"] == row["A_parent"]) &
                    (all_isotopes_in_chain["Z"] == row["Z_parent"]) &
                    (all_isotopes_in_chain["Elevel"] == row["Elevel"])
                ].index[0]

                child = all_isotopes_in_chain[
                    (all_isotopes_in_chain["A"] == row["A_child"]) &
                    (all_isotopes_in_chain["Z"] == row["Z_child"]) &
                    (all_isotopes_in_chain["Elevel"] == row["Elevel_child"])
                ]
                
                # Skip if no data for child
                if len(child) == 0:
                    continue

                child_index = child.index[0]

                if row["Total A,BM,BP,EC Intensity"] >= 100:
                    decay_branching = (
                        float(row["Intensity"]) / 
                        float(row["Total A,BM,BP,EC Intensity"])
                    )
                else:
                    decay_branching = float(row["Intensity"]) / 100
                
                decay_rate = decay_branching * np.log(2) / row["T1/2 (num)"]
                
                if decay_matrix[child_index, parent_index] == 0:
                    decay_matrix[child_index, parent_index] = decay_rate
                else:
                    decay_matrix[child_index, parent_index] += decay_rate

        # Convert to CSR format for CRAM48
        decay_matrix = sp.csr_matrix(np.nan_to_num(decay_matrix))
        
        return decay_matrix, all_isotopes_in_chain

    
    def decay_single_isotope(self, A, Z, Elevel, amount_gram, time_seconds):
        """Decay a single isotope using CRAM48 integration.
        
        Parameters
        ----------
        A : int
            Mass number
        Z : int
            Atomic number
        Elevel : float
            Energy level
        amount_gram : float
            Initial amount in grams
        time_seconds : float or list
            Decay time(s) in seconds
            
        Returns
        -------
        pd.DataFrame
            Decay products at specified time(s)
        """
        number_of_atoms = (amount_gram / A) * Avogadro
        
        # Check if decay data exists
        if len(
            self.data.ensdf[
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["T1/2 (num)"].unique()
        ) == 0:
            return pd.DataFrame(columns=[
                "A", "Z", "Element", "Elevel", "T1/2 (num)", 
                "Amount (number atoms)", "Decay Time (sec)", "Amount (gram)"
            ])

        # Get or generate decay matrix
        if (A, Z, Elevel) in self.decay_matrices_cache.keys():
            decay_matrix, only_parents = self.decay_matrices_cache[(A, Z, Elevel)]
        else:
            nuclide_chain = self.get_total_nuclide_chain(A, Z, Elevel)
            decay_matrix, only_parents = self.generate_decay_matrix(nuclide_chain)
            self.decay_matrices_cache[(A, Z, Elevel)] = (decay_matrix, only_parents)

        def perform_decay(time):
            N0 = np.zeros(len(only_parents))
            N0[0] = 1
            NT = openmc.deplete.cram.CRAM48(decay_matrix, N0, time)
            result = only_parents.copy()
            result["Amount (number atoms)"] = NT * number_of_atoms
            result["Decay Time (sec)"] = time
            result["Amount (gram)"] = (
                result["Amount (number atoms)"] / Avogadro * A
            )
            return result

        if isinstance(time_seconds, float):
            return perform_decay(time_seconds)
        elif isinstance(time_seconds, list):
            return pd.concat(
                [perform_decay(time) for time in time_seconds], 
                ignore_index=True
            )
        else:
            raise ValueError("time_seconds must be a float or a list of floats")

 
    @staticmethod   
    def combine_all_decay_products(all_decay_products):
        """
        Combine decay products from multiple isotopes.
        
        When decaying several isotopes, some of them can have the same children.
        This method sums them up so there is only one row per isotope per decay time.
        
        Parameters
        ----------
        all_decay_products : pd.DataFrame
            Concatenated decay results from multiple isotopes
            
        Returns
        -------
        pd.DataFrame
            Combined results with summed amounts
        """
        all_decay_products["full name"] = (
            all_decay_products["Element"] + 
            all_decay_products["A"].astype(int).astype(str) + 
            "-" + 
            all_decay_products["Elevel"].astype(str)
        )

        gb_amount_gram = all_decay_products.groupby(
            ["full name", "Decay Time (sec)"]
        )["Amount (gram)"].sum()
        
        gb_amount_number = all_decay_products.groupby(
            ["full name", "Decay Time (sec)"]
        )["Amount (number atoms)"].sum()

        list_total_gram = []
        list_total_number_atoms = []

        for _, row in all_decay_products.iterrows():
            key = (row["full name"], row["Decay Time (sec)"])
            list_total_gram.append(gb_amount_gram[key])
            list_total_number_atoms.append(gb_amount_number[key])

        all_decay_products["Total amount (number atoms)"] = list_total_number_atoms
        all_decay_products["Total amount (gram)"] = list_total_gram

        all_decay_products = all_decay_products.drop_duplicates(
            ["full name", "Decay Time (sec)"]
        ).reset_index(drop=True)
        
        all_decay_products = all_decay_products.drop(
            columns=["Amount (gram)", "Amount (number atoms)"]
        )

        return all_decay_products

    
    def decay_isotopic_mixture(self, isotopic_mixture):
        """
        Decay an isotopic mixture.
        
        Decays each isotope in the mixture and combines the results,
        summing up duplicate decay products.
        
        Parameters
        ----------
        isotopic_mixture : pd.DataFrame
            DataFrame with columns: A, Z, Elevel, Amount (gram), Decay_time (sec)
            
        Returns
        -------
        pd.DataFrame
            Combined decay products with totals
        """
        list_all_decay_results = []

        for _, row in isotopic_mixture.iterrows():
            decay_results = self.decay_single_isotope(
                row["A"], 
                row["Z"], 
                row["Elevel"], 
                row["Amount (gram)"], 
                row["Decay_time (sec)"]
            )
            list_all_decay_results.append(decay_results)

        if len(list_all_decay_results) == 0:
            columns = [
                'A', 'Element', 'Z', 'T1/2 (num)', 'Elevel',
                'Amount (number atoms)', 'Decay Time (sec)', 'Amount (gram)', 
                'full name', 'Total amount (number atoms)', 'Total amount (gram)', 
                'A decay start', 'Z decay start', 'Par. Elevel decay start'
            ]
            all_decay_products = pd.DataFrame(columns=columns)
            return all_decay_products

        all_decay_products = pd.concat(list_all_decay_results)
        all_decay_products = self.combine_all_decay_products(all_decay_products)
        return all_decay_products


    def clear_cache(self):
        """Clear decay matrix cache."""
        self.decay_matrices_cache = {}