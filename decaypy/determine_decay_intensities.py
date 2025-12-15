import numpy as np
import pandas as pd


class DecayIntensities:
    """
    Handles all intensity and branching ratio calculations.
    """
    
    def __init__(self, data, decay_instance, helper):
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
        self.helper = helper

        # Cache for intensity calculations
        self.total_decay_intensity_cache = {}
        self.EC_decay_intensity_cache = {}
    
    
    def return_total_decay_intensity(self, A, Z, Elevel):
        """Calculate total A, BM, BP, EC intensity for a nuclide.
        
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
            Total decay intensity
        """
        if (A, Z, Elevel) in self.total_decay_intensity_cache.keys():
            return self.total_decay_intensity_cache[(A, Z, Elevel)]

        int_A = float(np.sum(
            self.data.ensdf[
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Radiation"] == "A") & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["Rad Int."]
        ))
        
        int_BM = float(np.sum(
            self.data.ensdf[
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Radiation"] == "BM") & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["Rad Int."]
        ))
        
        int_BP = float(np.sum(
            self.data.ensdf[
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Radiation"] == "BP") & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["Rad Int."]
        ))

        total = int_A + int_BM + int_BP
        self.total_decay_intensity_cache[(A, Z, Elevel)] = total
        return total

    
    def return_total_fission_intensity(self, A, Z, Elevel):
        """Calculate total SF and FF intensity for a nuclide.
        
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
            Total fission intensity
        """
        int_SF = float(np.sum(
            self.data.ensdf[
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Radiation"] == "SF") & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["Rad Int."]
        ))
        
        int_FF = float(np.sum(
            self.data.ensdf[
                (self.data.ensdf["Z"] == Z) & 
                (self.data.ensdf["A"] == A) & 
                (self.data.ensdf["Radiation"] == "FF") & 
                (self.data.ensdf["Par. Elevel"] == Elevel)
            ]["Rad Int."]
        ))

        return int_SF + int_FF

    
    def return_EC_decay_intensity(self, A, Z, Elevel):
        """
        Calculate EC (electron capture) decay intensity.
        
        Handles floating point precision issues in the nuclear wallet cards
        where different Elevels might be represented identically.
        
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
            EC decay intensity (branching ratio)
        """
        if (A, Z, Elevel) in self.EC_decay_intensity_cache.keys():
            return self.EC_decay_intensity_cache[(A, Z, Elevel)]

        EC_branching = self.data.wallet_cards[
            (self.data.wallet_cards["A"] == A) & 
            (self.data.wallet_cards["Z"] == Z)
        ]
        
        # Account for rounding errors
        EC_branching = EC_branching[
            (EC_branching["Energy"] <= 1.005 * Elevel) & 
            (EC_branching["Energy"] >= 0.995 * Elevel)
        ]
        EC_branching = EC_branching[
            (EC_branching["Dec Mode"] == "EC")
        ]

        if len(EC_branching) == 1:
            branching = float(EC_branching["Branching (%)"].iloc[0])
            self.EC_decay_intensity_cache[(A, Z, Elevel)] = branching
            return branching

        if len(EC_branching) > 1:
            nndc_half_life = self.helper.get_from_nuclide_Elevel_half_life(
                A, Z, Elevel
            )
            EC_branching["Difference half-life"] = abs(
                EC_branching["T1/2 (seconds)"] - nndc_half_life
            )
            closest_branching = EC_branching.sort_values(
                by="Difference half-life"
            )["Branching (%)"].iloc[0]
            self.EC_decay_intensity_cache[(A, Z, Elevel)] = float(
                closest_branching
            )
            return float(closest_branching)

        self.EC_decay_intensity_cache[(A, Z, Elevel)] = 0.0
        return 0.0

    
    def clear_cache(self):
        """Clear all intensity caches."""
        self.total_decay_intensity_cache = {}
        self.EC_decay_intensity_cache = {}