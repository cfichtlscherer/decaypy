# nuclide_data_helper.py
import numpy as np
import pandas as pd
from importlib.resources import files
import openmc

class NuclideDataHelper:
    """
    Utility functions for querying nuclide data.
    Centralizes all data lookup and helper methods.
    """
    
    def __init__(self, data, decay_instance):
        """
        Parameters
        ----------
        data : DecayData
            The loaded decay data object
        """
        self.data = data
        self.from_nuclide_Elevel_half_life_cache = {}
        self.decay = decay_instance


    def get_from_nuclide_Elevel_half_life(self, A, Z, Elevel):
        """
        Retrieves the half-life value for a specific nuclide based on its atomic mass number (A), proton number (Z), and excited state energy level (Elevel).

        Parameters:
            A (int): The atomic mass number of the nuclide.
            Z (int): The proton number of the nuclide.
            Elevel (float): The excited state energy level of the nuclide.

        Returns:
            float: The half-life value of the nuclide in seconds.

        Raises:
            ValueError: If there are multiple half-life values for the nuclide, or if there are no half-life values for the nuclide.

        """
        if (A, Z, Elevel) in self.from_nuclide_Elevel_half_life_cache.keys():
            return self.from_nuclide_Elevel_half_life_cache[(A, Z, Elevel)]

        matching_rows = self.data.ensdf[(self.data.ensdf["A"] == A) & (self.data.ensdf["Z"] == Z) & (self.data.ensdf["Par. Elevel"] == Elevel)]
        unique_half_lives = matching_rows["T1/2 (num)"].unique()

        if len(unique_half_lives) > 1:
            raise ValueError(f"There are more than one half-life for the nuclide with A={A}, Z={Z}, Elevel={Elevel}.")
        elif len(unique_half_lives) == 0:
            raise ValueError(f"There is no half-life for the nuclide with A={A}, Z={Z}, Elevel={Elevel}.")
        else:
            hl_child = float(unique_half_lives[0])

        self.from_nuclide_Elevel_half_life_cache[(A, Z, Elevel)] = hl_child
        return hl_child


    def return_all_half_lifes_or_Elevel_for_a_nuclide(self, A, Z, return_hl=True):
        """Get all half-lives or energy levels for a nuclide.
        
        Parameters
        ----------
        A : int
            Mass number
        Z : int
            Atomic number
        return_hl : bool
            If True, return half-lives. If False, return energy levels.
            
        Returns
        -------
        np.ndarray
            Sorted unique half-lives or energy levels
        """
        half_lifes = np.sort(
            self.data.ensdf[
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Z"] == Z)
            ]["T1/2 (num)"].unique()
        )
        Elevel = np.sort(
            self.data.ensdf[
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Z"] == Z)
            ]["Par. Elevel"].unique()
        )

        if return_hl:
            return half_lifes
        else:
            return Elevel

    
    def get_closest_or_next_lowest(self, different_Elevel_of_child, energy_difference):
        """
        Find the closest energy level or next lowest if difference > 1%.
        
        The energy level of the child cannot be greater than the Q-value
        minus the radiation energy. Therefore, the closest energy level is
        the one that is smaller than or equal to the energy difference. If
        the difference is smaller than 1%, we assume there is a rounding
        error between the different datasets and take the closest value.
        Otherwise, we take the next lowest energy level. If there is no next
        lower energy level we take the closest value again.
        
        Parameters
        ----------
        different_Elevel_of_child : np.ndarray
            Available energy levels
        energy_difference : float
            Target energy to match
            
        Returns
        -------
        float
            Matched energy level
        """
        closest_value = min(
            different_Elevel_of_child, 
            key=lambda x: abs(x - energy_difference)
        )
        
        if np.isclose(closest_value, energy_difference, rtol=1e-3, atol=0):
            return closest_value
        
        else:
            lower_values = [
                level for level in different_Elevel_of_child 
                if level <= energy_difference
            ]
            if lower_values:
                return max(lower_values)
            else:
                return closest_value

    
    def from_endf_name_to_A_Z_Elevel(self, name):
        """
        Convert ENDF name format to (A, Z, Elevel).
        
        The ENDF FY data is given by names like 'Se79_m1' here this is
        brought into the (A, Z, Elevel) format.
        
        Parameters
        ----------
        name : str
            ENDF name (e.g., 'Se79_m1')
            
        Returns
        -------
        tuple
            (A, Z, Elevel) or ("-", "-", "-") if no data
        """
        A = float(''.join(filter(str.isdigit, name.split("_")[0])))
        iso_name = ''.join(filter(str.isalpha, name.split("_")[0]))
        Z = self.data.isotope_name_list.index(iso_name)
        
        if "_" not in name:
            Elevel = 0.0
            return ((A, Z, Elevel))
        else:
            all_Elevels = np.sort(
                self.data.ensdf[
                    (self.data.ensdf["A"] == A) & 
                    (self.data.ensdf["Z"] == Z)
                ]["Par. Elevel"].unique()
            )
            if len(all_Elevels) == 0:
                # there is no data on the SF product in the ENSDF data
                return (("-", "-", "-"))
            elif len(all_Elevels) == 1:
                Elevel = all_Elevels[0]
            else:
                Elevel = all_Elevels[1]
            return ((A, Z, Elevel))

    
    def return_q_value(self, A, Z, child):
        """
        Get Q-value for a decay.
        
        Parameters
        ----------
        A : int
            Mass number of parent
        Z : int
            Atomic number of parent
        child : tuple
            Child tuple (includes decay mode at index 4)
            
        Returns
        -------
        float
            Q-value or 0.0 if not available
        """
        decay_mode = child[4]
        n = int(A - Z)
        key = (n, int(Z))

        if decay_mode == "A":
            q_value = self.data.q_alpha.get(key, 0.0)
        elif decay_mode == "BM":
            q_value = self.data.q_beta_minus.get(key, 0.0)
        elif decay_mode == "BP":
            q_value = self.data.q_beta_plus.get(key, 0.0)
        elif decay_mode == "EC":
            q_value = self.data.q_ec.get(key, 0.0)
        elif decay_mode in ["SF", "TF", "FF", "DTF"]:
            q_value = 0.0
        else:
            q_value = 0.0

        return q_value

    
    def return_fission_fractions(self, A, Z, Elevel):
        """
        Get the fission fraction for a nuclide.
        
        This function reads out the fission fraction (the chance of performing 
        SF when decaying) of the nuclide from the Nuclear Wallet Cards.
        The fission rate is given in proportion not in percent.
        
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
        float
            Fission fraction (0.0 if not available)
        """
        try:
            fission_fraction = float(
                self.data.wallet_cards[
                    (self.data.wallet_cards["A"] == A) & 
                    (self.data.wallet_cards["Z"] == Z) & 
                    (self.data.wallet_cards["Energy"] == Elevel) & 
                    (self.data.wallet_cards["Dec Mode"] == "SF")
                ]["Branching (%)"].values[0]
            ) / 100
        except IndexError:
            fission_fraction = 0.0

        return fission_fraction
    

    def return_fission_yields(self, A, Z, Elevel, fission_type):
        """ Returns fission yields.
        Read out fission yields from ENDF/B-VIII.0 data with the OpenMC
        FissionProductYields class. If no SF yield is given, it is approximated with
        the thermal FY.
        """

        if fission_type not in ["SF", "TF", "FF", "DTF"]:
            raise ValueError("Invalid fission type. 'fission_type' can be set to: SF, TF, FF, DTF")

        # 3-tuples of (A, Z, Elevel)
        SFY_available = [(238, 92, 0.0), (244, 96, 0.0), (246, 96, 0.0),
                         (248, 96, 0.0), (250, 98, 0.0), (252, 98, 0.0),
                         (253, 99, 0.0), (254, 100, 0.0), (256, 100, 0.0)]
        neutron_induced_FY_available = [(227, 90, 0.0), (229, 90, 0.0), (232, 90, 0.0),
                                        (231, 91, 0.0), (232, 92, 0.0), (233, 92, 0.0),
                                        (234, 92, 0.0), (235, 92, 0.0), (236, 92, 0.0),
                                        (237, 92, 0.0), (238, 92, 0.0), (237, 93, 0.0),
                                        (238, 93, 0.0), (238, 94, 0.0), (239, 94, 0.0),
                                        (240, 94, 0.0), (241, 94, 0.0), (242, 94, 0.0),
                                        (241, 95, 0.0), (242, 95, 48.63), (243, 95, 0.0),
                                        (242, 96, 0.0), (243, 96, 0.0), (244, 96, 0.0),
                                        (245, 96, 0.0), (246, 96, 0.0), (248, 96, 0.0),
                                        (249, 98, 0.0), (251, 98, 0.0), (254, 99, 0.0),
                                        (255, 100, 0.0)]

        nuclide_name = self.data.isotope_name_list[int(Z)]
        Z_name = str(int(Z)).zfill(3)

        if ((A, Z, Elevel) in SFY_available) and (fission_type=="SF"):
            file_name = f"sfy-version.VIII.1/sfy-{Z_name}_{nuclide_name}_{str(int(A))}.endf"
        elif (A, Z, Elevel) in neutron_induced_FY_available:
            if Elevel > 0.0:
                file_name = f"nfy-version.VIII.1/nfy-{Z_name}_{nuclide_name}_{str(int(A))}m1.endf"
            else:
                file_name = f"nfy-version.VIII.1/nfy-{Z_name}_{nuclide_name}_{str(int(A))}.endf"
        else:
            return []

        if fission_type == "SF":
            fission_fraction = self.return_fission_fractions(A, Z, Elevel)
        else:
            fission_fraction = 1

        FY_data = openmc.data.FissionProductYields(files("decaypy.data.ENDF_FY").joinpath(file_name))

        fission_keys = {"SF": 0, "TF": 0, "FF": 1, "DTF": 2}

        FY_list = []
        if len(list(FY_data.independent)) == 1:
            fission_type = "TF"
        for isotope in FY_data.independent[fission_keys[fission_type]].keys():
            iso_fy = FY_data.independent[fission_keys[fission_type]][isotope].nominal_value
            if iso_fy > 0.0:
                A_Z_Elevel = self.from_endf_name_to_A_Z_Elevel(isotope)
                if A_Z_Elevel != ("-", "-", "-"):
                    FY_list += [(A_Z_Elevel, iso_fy * fission_fraction)]

        FY_list_sorted = sorted(FY_list, key=lambda x: x[1])[::-1]
        
        if self.decay.fission_yield_reduction == False:
            return FY_list_sorted
        return FY_list_sorted[:self.decay.fission_yield_reduction]

    def clear_cache(self):
        """Clear decay matrix cache."""
        self.from_nuclide_Elevel_half_life_cache = {}