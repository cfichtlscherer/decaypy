import numpy as np
from scipy.constants import Avogadro


# the Watt-distribution parameters a and b for spontaneous fission neutrons
# source: UCRL-AR-228518-REV-1, table 5 & LA-13422-M, table 5
SF_watt_constants = {"U235m0": (1.29080, 4.85231),
                     "U238m0": (1.54245, 6.81057),
                     "Pu238m0": (1.17948, 4.16933),
                     "Pu239m0": (1.12963, 3.80269),
                     "Pu240m0": (1.25797, 4.68927),
                     "Pu241m0": (1.18698, 4.15150),
                     "Pu242m0": (1.22078, 4.36668),
                     "Am241m0": (1.07179, 3.46195),
                     "Cf252m0": (0.847458, 1.03419)}


# the Watt-distribution parameters a for fast (2MeV) fission
# source: calculated with the values from table 6 and equation 3 of 3UCRL-AR-228518-REV-1
fast_fission_watt_constant_a = {"Pu238m0": 0.8697191336,
                                "Pu239m0": 0.8674254568,
                                "Pu240m0": 0.8687426148,
                                "Pu241m0": 0.8615824056,
                                "Pu242m0": 0.869396624}


# spontaneous fission branching ratios (values given as fractions)
# source: Kütt, Simulation of Neutron Multiplicity Measurements using Geant4, 2016 (table 4.2).
# https://tuprints.ulb.tu-darmstadt.de/5621/ 
branching_ratios = {"U235m0": 7.0e-11,
                    "U238m0": 5.45e-7,
                    "Pu238m0": 1.9e-9,
                    "Pu239m0": 3.1e-12,
                    "Pu240m0": 5.7e-8,
                    "Pu241m0": 2e-16,
                    "Pu242m0": 5.5e-6,
                    "Am241m0": 3.6e-12,
                    "Cm242m0": 0.6e-7,
                    "Cm244m0": 1.37e-6,
                    "Cf252m0": 0.0309}


# half-lifes (seconds)
# source: IAEA livechart Table of Nuclides (mainly ENSDF data)
# https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html (2022-11-13 So 14:41)
half_lives = {"U235m0": 2.221e+16,
              "U238m0": 1.41e+17,
              "Pu238m0": 2768000000.0,
              "Pu239m0": 760900000000.0,
              "Pu240m0": 207050000000.0,
              "Pu241m0": 451000000.0,
              "Pu242m0": 11770000000000.0,
              "Am241m0": 13650000000.0,
              "Cm242m0": 14070000.0,
              "Cm244m0": 571500000.0,
              "Cf252m0": 83470000.0}


# average neutron multiplicities per spontaneous fission
# source: Kütt, Simulation of Neutron Multiplicity Measurements using Geant4, 2016 (table 4.3).
# https://tuprints.ulb.tu-darmstadt.de/5621/ 
multiplicities = {"U235m0": 1.86,
                  "U238m0": 1.99,
                  "Pu238m0": 2.187,
                  "Pu239m0": 2.16,
                  "Pu240m0": 2.154,
                  "Pu241m0": 2.25,
                  "Pu242m0": 2.149,
                  "Am241m0": 3.22,
                  "Cm242m0": 2.54,
                  "Cm244m0": 2.72,
                  "Cf252m0": 3.757}


# atomic weights (in micro AMU)
# source: IAEA livechart Table of Nuclides (mainly ENSDF data)
# https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html (2022-11-13 So 14:41)
atomic_weights = {"U235m0": 235043928.1,
                  "U238m0": 238050786.9,
                  "Pu238m0": 238049558.2,
                  "Pu239m0": 239052161.6,
                  "Pu240m0": 240053811.7,
                  "Pu241m0": 241056849.7,
                  "Pu242m0": 242058741.0,
                  "Am241m0": 241056827.3,
                  "Cf252m0": 252081626.5}


def watt_distribution(a, b, E):
    """
    evaluations of Watt-distribution at the value E
    result is the probability for value E

    a : isotope specific value for the Watt-distribution
    b : isotope specific value for the Watt-distribution
    E : Energy of evaluation [MeV]
    """

    c = ((np.pi*b)/(4*a))**0.5 * ((np.exp(b/(4*a)))/a)
    w = c * np.exp(-a*E) * np.sinh((b*E)**0.5)

    return w


def number_of_SF_reactions(isotope, amount):
    """
    input: amount of an isotope
    output: number of  SF rate

    isotope : name of the isotope eg Pu240m0
    amount : amount of this isotope in gram
    """

    if isotope not in branching_ratios.keys():
        raise ValueError("No data for isotope in branching_ratios dictionary")
    if isotope not in half_lives.keys():
        raise ValueError("No data for isotope in half-lifes dictionary")
    if isotope not in multiplicities.keys():
        raise ValueError("No data for isotope in multiplicities dictionary")
    if isotope not in atomic_weights.keys():
        raise ValueError("No data for isotope in atomic weights dictionary")

    number_of_atoms = (amount / (atomic_weights[isotope] * 1e-6)) * Avogadro
    decay_constant = np.log(2) / (half_lives[isotope])
    decays_per_second = decay_constant * number_of_atoms
    sf_second = branching_ratios[isotope] * decays_per_second

    return sf_second


def number_produced_SF_neutrons(isotope, amount):
    """
    input: amount of an isotope
    output: number of produced neutrons per second by SF

    isotope : name of the isotope eg Pu240m0
    amount : amount of this isotope in gram
    """

    sf_second = number_of_SF_reactions(isotope, amount)
    produced_neutrons_second = sf_second * multiplicities[isotope]

    return produced_neutrons_second
