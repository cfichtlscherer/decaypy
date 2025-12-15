import unittest
import pandas as pd
import numpy as np
import openmc
import math as math
from pathlib import Path

from decaypy.load_data import _cleanstring
from decaypy.load_data import load_decay_data
from decaypy.nuclide_data_helper import NuclideDataHelper
from decaypy.decay_and_emissions import Decay
from decaypy.neutron_emissions import number_produced_SF_neutrons, number_of_SF_reactions

DATA_DIR = Path(__file__).parent.parent / "decaypy" / "data"
ENDF_FY_DIR = DATA_DIR / "ENDF_FY"


class TestDecay(unittest.TestCase):
    """
    This class contains test cases for the entire module.
    """

    def setUp(self):
        self.decay = Decay()
        self.data = load_decay_data(include_endf_decay_data=False)
        self.helper = NuclideDataHelper(self.data, self.decay)
        

    def test_cleanstring(self):
        # Test the cleanstring method
        self.assertEqual(_cleanstring("123"), 123.0)
        self.assertEqual(_cleanstring("123.45"), 123.45)
        self.assertEqual(_cleanstring("abc"), "abc")
        self.assertEqual(_cleanstring("  abc  "), "abc")


    def test_return_fission_fractions(self):
        # Test the return_fission_fractions method
        # the fission rate is given in proportion not in percent
        fission_fraction = self.helper.return_fission_fractions(239, 94, 0.0)
        self.assertEqual(fission_fraction, 3e-12)
        fission_fraction = self.helper.return_fission_fractions(239, 95, 0.0)
        self.assertEqual(fission_fraction, 0.0)
        fission_fraction = self.helper.return_fission_fractions(250, 96, 0.0)
        self.assertEqual(fission_fraction, 0.74)


    def test_return_fission_yields(self):
        self.decay.fission_yield_reduction=1
        # SF of Pu239, check yield of most produced child
        fission_fraction = self.helper.return_fission_fractions(239, 94, 0.0)
        pu239_path = (ENDF_FY_DIR / "nfy-version.VIII.1" / "nfy-094_Pu_239.endf")
        FY_openmc_data = openmc.data.FissionProductYields(pu239_path)
        endf_fy_yield = FY_openmc_data.independent[0]["Zr100"].nominal_value
        fission_yields = self.helper.return_fission_yields(239, 94, 0.0, "SF")
        self.assertEqual(fission_yields[0], ((100.0, 40, 0.0), fission_fraction * endf_fy_yield))
        
        # FF of U235, check yield of most produced child, fission is induced so no fission fraction
        u235_path = (ENDF_FY_DIR / "nfy-version.VIII.1" / "nfy-092_U_235.endf")
        FY_openmc_data = openmc.data.FissionProductYields(u235_path)
        endf_fy_yield = FY_openmc_data.independent[1]["Te134"].nominal_value
        fission_yields = self.helper.return_fission_yields(235, 92, 0.0, "FF")
        self.assertEqual(fission_yields[0], ((134, 52, 0.0), endf_fy_yield))


    def test_decay_single_isotope_only_ff(self):
        self.decay.with_SF_decay=False
        fissions_per_second_per_atom = {(239, 94, 0.0): (1e-10, "FF")}
        self.decay.induced_fission_rates = fissions_per_second_per_atom
        # while JEFF3.1.1 data suggests that I135 is the most produced fast
        # fission product of Pu239, ENDF data suggests Mo104 as the most
        # produced fission product.
        #
        # In the ENDF data Mo104 has a fast fission yield of 0.04613114
        # => 1e-10 * 0.04613114 = 4.613114e-12 produced Mo104 atoms per second per pu-239 atom
        # Mo104 has a half-life of 60s => np.log(2)/60 = 0.011552 Mo104 atoms decay per second
        # there should be a Secular Equilibrium: lambda1 * N1 = lambda2 * N2
        # 4.613114e-12 * Avogadro / 0.011552 = 240484953687038.1 = 2.4e14 atoms of Mo104 to be expect
        self.decay.fission_yield_reduction=1
        decay_results = self.decay.decay_single_isotope(239, 94, 0.0, 239, 24*3600.0)
        Mo104_number_of_atoms = float(decay_results.loc[(decay_results["A"] == 104) & (decay_results["Z"] == 42) & (decay_results["Elevel"] == 0.0), "Amount (number atoms)"].values[0])
        self.assertTrue(math.isclose(Mo104_number_of_atoms, 240484953687038.1, rel_tol=1e-3))


    def test_decay_single_isotope_only_sf(self):
        self.decay.with_SF_decay=True
        self.decay.induced_fission_rates = {}
        # the fission fraction of Pu240 is 5.7E-8 per decay.
        # the most common SF decay product is Te134 with a yield of 0.0531169 (ENDF VIII.1)
        # =>  (np.log(2) / 2.070500e+11) * 5.7E-8 * 0.0531169 = 1.0135794639361602e-20
        # every second 1.0135794639361602e-20 Te134 atoms are produced per pu240 atom
        # Te134 has a half-life of 2.508000e+03. 
        # Assuming 1 mole of Pu240 (i.e. Avogadro atoms)
        # there is a Secular Equilibrium: lambda1 * N1 = lambda2 * N2
        # Avogadro * 1.0135794639361602e-20 / (np.log(2) / 2.508000e+03) = 22085680.04479457
        # there should be about 2.2085680e7 atoms of Te134
        
        self.decay.fission_yield_reduction=1
        decay_results = self.decay.decay_single_isotope(240, 94, 0.0, 240, 24*3600.0)
        Te134_number_of_atoms = float(decay_results.loc[(decay_results["A"] == 134) & (decay_results["Z"] == 52) & (decay_results["Elevel"] == 0.0), "Amount (number atoms)"].values[0])
        self.assertTrue(math.isclose(Te134_number_of_atoms, 22085680.04479457, rel_tol=1e-3))
        # the code currently calculates 24606722.776747577 atoms of Te134, which is about 11% higher than expected.


    def test_decay_single_isotope_only_sf_2(self):
        self.decay.with_SF_decay=True
        self.decay.induced_fission_rates = {}
        # Every decay of Pu239 produces 1.434942e-13 atoms of Zr100 per decay
        # 1.2882009e-13
        # Zr100 has a half-life of 7.1 seconds
        # In a Secular Equilibrium there should be:
        # (np.log(2)/ 7.609000e+11)* 1.434942e-13 * Avogadro / (np.log(2)/7.1) = 0.8063359339689188
        # atoms of Zr100
        self.decay.fission_yield_reduction = 1
        decay_results = self.decay.decay_single_isotope(239, 94, 0.0, 239, 24*3600.0)
        Zr100_number_of_atoms = float(decay_results.loc[(decay_results["A"] == 100) & (decay_results["Z"] == 40) & (decay_results["Elevel"] == 0.0), "Amount (number atoms)"].values[0])
        self.assertTrue(math.isclose(Zr100_number_of_atoms, 0.8063359339689188, rel_tol=1e-3))


    def test_decay_single_isotope_sf_and_ff(self):
        #use ff and sf together. The production rates are still separated, as
        #we have only 1 fission yield product.
        #we check them separately.
        self.decay.with_SF_decay = True
        fissions_per_second_per_atom = {(239, 94, 0.0): (1e-10, "FF")}
        self.decay.induced_fission_rates = fissions_per_second_per_atom
        self.decay.fission_yield_reduction = 1
        decay_results = self.decay.decay_single_isotope(239, 94, 0.0, 239, 24*3600.0)
        Zr100_number_of_atoms = float(decay_results.loc[(decay_results["A"] == 100) & (decay_results["Z"] == 40) & (decay_results["Elevel"] == 0.0), "Amount (number atoms)"].values[0])
        Mo104_number_of_atoms = float(decay_results.loc[(decay_results["A"] == 104) & (decay_results["Z"] == 42) & (decay_results["Elevel"] == 0.0), "Amount (number atoms)"].values[0])
        self.assertTrue(math.isclose(Zr100_number_of_atoms, 0.8063359339689188, rel_tol=1e-3))
        self.assertTrue(math.isclose(Mo104_number_of_atoms, 240484953687038.1, rel_tol=1e-3))


    def test_decay_single_isotope_cs137(self):
        decay_results = self.decay.decay_single_isotope(137, 55, 0.0, 137, 9.493000e+08)
        gram_cs_137 = float(decay_results.loc[0, "Amount (gram)"])
        self.assertEqual(np.round(gram_cs_137,8), 137/2)


    def test_decay_single_isotope_pu241(self):
        self.decay.with_SF_decay=True
        self.decay.induced_fission_rates = {(241, 94, 0.0): (1e-10, "FF")}
        decay_results = self.decay.decay_single_isotope(241, 94, 0.0, 241, 2*4.51e8)
        # the decay constant (including FF) of pu241 is:
        # (np.log(2) / 4.51e8) + 1e-10 = 1.636911708558637e-09
        # after 2*4.51e8 seconds the amount of pu241 should be:
        # 241 * np.exp(-1.636911708558637e-09 * 2*4.51e8) = 55.04 gram
        gram_pu_241 = float(decay_results.loc[0, "Amount (gram)"])
        self.assertTrue(math.isclose(gram_pu_241, 55.053, rel_tol=1e-3))        
        # the amount of am241 should be (Bateman equations):
        # f = branching ration pu-241->am-241 ((np.log(2) / 4.51e8) / ((np.log(2) / 4.51e8) + 1e-10))
        # l1 = decay_constant_pu241
        # l2 = decay_constant_am241
        # N0 = amount_pu241(t=0)
        # T = decay time
        # ((f * l1 * N0) / (l2 - l1)) * (np.exp(-l1 * T) - (np.exp(-l2 * T)))
        # = 169.72161 gram of Am241
        am_241_row = decay_results[(decay_results["Element"] == "AM") & 
                                   (decay_results["A"] == 241) & 
                                   (decay_results["Elevel"] == 0.0)]
        gram_am_241 = float(am_241_row["Amount (gram)"].values[0])
        self.assertTrue(math.isclose(gram_am_241, 169.72161, rel_tol=1e-3))        


    def test_no_negative_material_amounts_pu240(self):
        self.decay.with_SF_decay=True
        self.decay.induced_fission_rates = {(240, 94, 0.0): (1e-10, "FF")}
        decay_results = self.decay.decay_single_isotope(241, 94, 0.0, 241, 10.0*365*24*3600)

        # Check that there are no negative amounts in the decay results
        negative_amounts = decay_results[decay_results["Amount (gram)"] < 0]
        self.assertTrue(negative_amounts.empty, "There are decay results with negative amounts (gram).")

    
    def test_multiple_decay_steps_single_isotope(self):
        decay_results = self.decay.decay_single_isotope(137, 55, 0.0, 137, [0.0, 9.493000e+08, 10.0**100])
        cs137_amounts_gram = decay_results[(decay_results["Element"] == "CS") & (decay_results["A"] == 137)]["Amount (gram)"].tolist()
        self.assertEqual(np.round(cs137_amounts_gram[0],8), 137)
        self.assertEqual(np.round(cs137_amounts_gram[1],8), 137/2)
        self.assertEqual(np.round(cs137_amounts_gram[2],8), 0.0)

    
    def test_multiple_decay_steps_isotopic_mixture(self):
        iso_mixture = pd.DataFrame({'A': [60, 137],                                      
                                    'Z': [27, 55],                                           
                                    'Elevel': [0.0, 0.0],                                 
                                    'Amount (gram)': [60, 137],                  
                                    'Decay_time (sec)': 2*[[1.663000e+08, 9.493000e+08]]})    
        decay_results = self.decay.decay_isotopic_mixture(iso_mixture)
        co60_amount_1_66 = decay_results[(decay_results["Element"] == "CO") & (decay_results["A"] == 60) & (decay_results["Decay Time (sec)"] == 1.663000e+08)]["Total amount (gram)"].values[0]
    
        cs137_amount_9_49 = decay_results[(decay_results["Element"] == "CS") & (decay_results["A"] == 137) & (decay_results["Decay Time (sec)"] == 9.493000e+08)]["Total amount (gram)"].values[0]

        self.assertAlmostEqual(co60_amount_1_66, 30, places=2, msg="Co60 amount after 1.663000e+08 seconds is not as expected.")
        self.assertAlmostEqual(cs137_amount_9_49, 137/2, places=2, msg="Cs137 amount after 9.493000e+08 seconds is not as expected.")


    def test_EC_non_numeric_issue(self):
        # ONIX EC_branching["Energy"] contains string values instead of numeric values.
        # takes about 20 minutes, iterates over all nuclides
        # for the fast checks its turned off
        if False:
            self.decay.include_ENDF_decay_data=True
            data_path = DATA_DIR / "data_processed" / "big_df_all_nuclide_data_including_ONIX_ENDF.pickle"
            df = pd.read_pickle(data_path)
            grouped_df = df.groupby(["A", "Z", "Par. Elevel"])
            # check every nuclide
            for (A, Z, Par_Elevel), _ in grouped_df:
                self.decay.decay_single_isotope(A, Z, Par_Elevel, 1, 24*3600.0)


    def test_return_isotope_gamma_emissions(self):
        # starting with 1 mole of Cs137
        # after 30.04 years half of Cs137 should be decayed, resulting in 0.5 mole of Cs137
        # with a chance of about 94.6% Cs137 decays to Ba137m
        # after 30.04 years the resulting Cs137 is in an equilibrium with Ba137m
        # Ba137m has a half-life of 2.552 minutes
        # the amount of Ba137m can be calculated as follows:
        # N(Ba137m) = branching * (T12_Ba / T12_Cs) * N(Cs137)
        # 0.946 * ((2.552 * 60) / (30.04*365*24*3600)) * 0.5 * Avogadro = 4.610161100223845e+16 atoms of Ba137m
        # Ba137m decays to Ba137 with the emission of a 661.7 keV gamma with 85.1% intensity
        # the number of created 661.7keV gammas per second is:
        # 4.610161100223845e+16 * (np.log(2) / (2.552*60)) * 0.899 = 187615767471526.7 gammas per second
        decay_results = self.decay.decay_single_isotope(137, 55, 0.0, 137, 30.04*365*24*3600.)
        number_of_atoms_ba137m = float(decay_results.loc[1, "Amount (number atoms)"])
        self.assertTrue(math.isclose(number_of_atoms_ba137m, 4.610161100223845e+16, rel_tol=1e-3))
        all_gamma_emissions = self.decay.return_gamma_emissions_isotopic_mixture(decay_results)
        gamma_emission_662 = float(all_gamma_emissions[(all_gamma_emissions["Rad Ene."] == 661.657)]["Total emissions 1/sec"])
        self.assertTrue(math.isclose(gamma_emission_662, 187615767471526.7, rel_tol=1e-3))

    
    def test_SF_rate(self):
        # Similar to:
        # Kütt, Simulation of Neutron Multiplicity Measurements using Geant4, 2016 (table 4.3).
        # https://tuprints.ulb.tu-darmstadt.de/5621/  
        # Assuming 1 gram of Cf252
        # Cf252 has a T12 of 2.646 years and a branching ratio of 3.09%
        # Meaning for every alpha-decay there is a 3.09% chance for a spontaneous fission
        # 0.0309 * (np.log(2) / (2.646*365*24*3600)) * (1/252) * Avogadro = 613623549860.1754 'SF per s'
        
        SF_reactions_1g_Cf252 = number_of_SF_reactions("Cf252m0", 1.0)
        self.assertTrue(math.isclose(SF_reactions_1g_Cf252, 613391643756.6758, rel_tol=1e-3))
        

    def test_produced_SF_neutrons(self):        
        # Similar to:
        # Kütt, Simulation of Neutron Multiplicity Measurements using Geant4, 2016 (table 4.3).
        # https://tuprints.ulb.tu-darmstadt.de/5621/  
        # Assuming 1 gram of Am241
        # Am241 has a T12 of 432.6 years, a branching ratio of 3.6e-10% and a SF yield of 3.22
        # 3.6e-12 * (np.log(2) / (432.6*365*24*3600)) * (1/241) * Avogadro * 3.22 = 1.471717897275418
        
        SF_neutrons_1g_Am241 = number_produced_SF_neutrons("Am241m0", 1.0)
        self.assertTrue(math.isclose(SF_neutrons_1g_Am241, 1.471717897275418, rel_tol=1e-3))


if __name__ == '__main__':
    unittest.main()
