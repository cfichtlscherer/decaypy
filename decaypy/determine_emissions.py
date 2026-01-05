# determine_emissions.py
import numpy as np
import pandas as pd
from scipy.constants import Avogadro


class GammaEmissions:
    """
    Handles all gamma emission calculations for nuclear decay chains.
    Separate module for cleaner code organization.
    """
    
    def __init__(self, data, decay_instance):
        """
        Parameters
        ----------
        data : DecayData
            The loaded decay data object
        decay_instance : Decay
            Reference to parent Decay class instance to access its methods
        """
        self.data = data
        self.decay = decay_instance  # Reference to parent Decay instance
    
    
    def return_isotope_gamma_emissions(self, A, Z, Elevel):
        """Returns the lines of the pandas dataframe with the gamma emissions
        
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
            DataFrame with gamma emission data
        """
        
        # In the following it is checked if gamma lines are shared from parents 
        # and children to avoid double counting
        # Including fission children here would interfere with this logic
        self.decay.induced_fission_rates = {}
        self.decay.include_sf = False
        self.decay.decays_cache = {}
        self.decay.from_nuclide_Elevel_half_life_cache = {}
        self.decay.total_decay_intensity_cache = {}
        self.decay.EC_decay_intensity_cache = {}

        gamma_emissions = self.data.ensdf[
            (self.data.ensdf["A"] == A) & 
            (self.data.ensdf["Z"] == Z) & 
            (self.data.ensdf["Par. Elevel"] == Elevel) & 
            (self.data.ensdf["Radiation"] == "G")
        ]
        
        # case1: no gamma emissions at all
        if len(gamma_emissions) == 0:
            return gamma_emissions
        else:
            all_gamma_emissions_pd_list = []

            # case2: (gamma emissions) & (isotope == daughter)
            isotope_name = str(int(A)) + gamma_emissions["Element"].unique()[0]
            gamma_emissions_deexcitation = gamma_emissions[
                gamma_emissions["Daughter"] == isotope_name
            ]
            all_gamma_emissions_pd_list.append(gamma_emissions_deexcitation)

            # case3: (gamma emissions) & (isotope != daughter)
            # we check the gamma lines of the direct children of that isotope
            # and take only those lines that are not produced by any child
            all_children = self.decay.get_children_one_generation(A, Z, Elevel)
            gamma_emissions_children_list = []

            for child in all_children:
                if np.isnan(child[0]):
                    # no data on child or "exotic" decays
                    continue
                    
                daughter_name = (
                    str(int(child[0])) + 
                    self.data.isotope_name_list[int(child[1])].upper()
                )
                
                if (isotope_name != daughter_name) and (
                    len(self.data.ensdf[
                        (self.data.ensdf["A"] == child[0]) & 
                        (self.data.ensdf["Z"] == child[1])
                    ]) > 0
                ):
                    all_Elevel_child, all_intensities = (
                        self.decay.return_all_reached_elevel(A, Z, Elevel, child)
                    )
                    
                    for Elevel_child in all_Elevel_child[all_intensities > 0.0]:
                        gamma_emissions_child = self.data.ensdf[
                            (self.data.ensdf["A"] == child[0]) & 
                            (self.data.ensdf["Z"] == child[1]) & 
                            (self.data.ensdf["Par. Elevel"] == Elevel_child) & 
                            (self.data.ensdf["Radiation"] == "G")
                        ]
                        gamma_emissions_children_list.append(gamma_emissions_child)

            if len(gamma_emissions_children_list) == 0:
                all_gamma_emissions_children = self.data.ensdf[
                    self.data.ensdf["A"] == 999
                ]
            else:
                all_gamma_emissions_children = pd.concat(
                    gamma_emissions_children_list
                )

            # Get all gammas that are in gamma_emissions but not in 
            # all_gamma_emissions_children
            gamma_emissions_not_in_children = gamma_emissions[
                ~gamma_emissions['Rad Ene.'].isin(
                    all_gamma_emissions_children['Rad Ene.']
                )
            ]
            all_gamma_emissions_pd_list.append(gamma_emissions_not_in_children)

            if len(all_gamma_emissions_pd_list) == 0:
                return self.data.ensdf[self.data.ensdf["A"] == 999]  # empty df

            else:
                all_gamma_emissions_pd = pd.concat(all_gamma_emissions_pd_list)
                # make sure we don't do any double counting
                all_gamma_emissions_pd = all_gamma_emissions_pd.drop_duplicates()

            return all_gamma_emissions_pd

    
    def return_gamma_emissions_isotopic_mixture(self, isotopic_mixture, 
                                               add_iso_cols=False):
        """
        Calculate gamma emissions for an isotopic mixture.
        
        Parameters
        ----------
        isotopic_mixture : pd.DataFrame
            DataFrame with columns: A, Z, Elevel, Total amount (gram)
        add_iso_cols : bool, optional
            If True, add parent isotope columns to output. Default: False
            
        Returns
        -------
        pd.DataFrame
            Grouped gamma emissions by energy with total emissions per second
        """
        
        # Exclude rows where Element is "Unknown" / no data on the isotope
        isotopic_mixture = isotopic_mixture.loc[
            isotopic_mixture["Element"] != "Unknown"
        ]

        all_gamma_emissions_mixture_list = []

        for _, row in isotopic_mixture.iterrows():
            all_gamma_emissions_pd = self.return_isotope_gamma_emissions(
                row["A"], row["Z"], row["Elevel"]
            )
            
            # Convert radiation intensity to float
            all_gamma_emissions_pd['Rad Int.'] = (
                all_gamma_emissions_pd['Rad Int.'].replace(
                    r'^\s*$', 0.0, regex=True
                )
            )
            all_gamma_emissions_pd['Rad Int.'] = (
                all_gamma_emissions_pd['Rad Int.'].astype(float)
            )
            
            # Calculate total emissions per second
            all_gamma_emissions_pd["Total emissions 1/sec"] = (
                (row["Total amount (gram)"] / row["A"] * Avogadro) *
                (np.log(2) / all_gamma_emissions_pd["T1/2 (num)"]) *
                (all_gamma_emissions_pd["Rad Int."] / 100.0)
            )
            
            if add_iso_cols:
                all_gamma_emissions_pd["Parent_A"] = row["A"]
                all_gamma_emissions_pd["Parent_Z"] = row["Z"]
                all_gamma_emissions_pd["Parent_Elevel"] = row["Elevel"]
                
            all_gamma_emissions_mixture_list.append(all_gamma_emissions_pd)

        all_gamma_emissions_mixture = pd.concat(all_gamma_emissions_mixture_list)
        
        # The same nuclide can be produced multiple times. 
        # Here we sum up the emissions of gammas of the same energy.
        if add_iso_cols:
            all_gamma_emissions_mixture_grouped = (
                all_gamma_emissions_mixture.groupby(["Rad Ene."]).agg(
                    {
                        "Total emissions 1/sec": "sum",
                        "T1/2 (num)": "first",
                        "Rad Int.": "first",
                        "Parent_A": "first",
                        "Parent_Z": "first",
                        "Parent_Elevel": "first"
                    }
                ).reset_index()
            )
        else:
            all_gamma_emissions_mixture_grouped = (
                all_gamma_emissions_mixture.groupby(["Rad Ene."]).agg(
                    {
                        "Total emissions 1/sec": "sum",
                        "T1/2 (num)": "first",
                        "Rad Int.": "first",
                    }
                ).reset_index()
            )

        return all_gamma_emissions_mixture_grouped
