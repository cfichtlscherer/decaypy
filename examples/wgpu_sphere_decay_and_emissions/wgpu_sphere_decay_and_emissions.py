"""
    Determining the decay and created emissions for a 3kg sphere of weapons-grade plutonium.
    This is a sub-critical system in which the effect of spontaneous fission and induced fission are both important.
    1) Determine the spontaneous fission neutron production rates for each isotope in the sphere.
    2) Use OpenMC to simulate the neutron transport and determine the induced fission rates
    3) Use decaypy to determine the decay products
    4) Use decaypy to determine the created emissions
    5) Use the OpenMCconnector to create an OpenMC material from the decayed composition
"""

import numpy as np
import pandas as pd
from scipy.constants import Avogadro
import openmc

from decaypy.decay_and_emissions import Decay
from decaypy.neutron_emissions import number_produced_SF_neutrons, watt_distribution, SF_watt_constants, fast_fission_watt_constant_a, atomic_weights
from decaypy.openmc_connector import inventory_to_openmc_material, apply_material_lines

# change these to your custom paths
openmc_path = "/home/cpf/apps/openmc/build/bin/openmc"
crosssection_path = "/home/cpf/all_openmc_xsections/official_libraries/endfb80_hdf5"

# 1) Determine the spontaneous fission neutron production rates for each isotope in the sphere.
wgpu_3013_fractions = [("Pu238m0", 0.0005), ("Pu239m0", 0.935), ("Pu240m0", 0.06), ("Pu241m0", 0.004), ("Pu242m0", 0.0005)]
gallium_content = 0.008
gallium_stabilized_plutonium_gram = 3000

neutron_rates = {}
material_grams ={}

for isotope, fraction in wgpu_3013_fractions:
    plutonium_isotope_grams = fraction * gallium_stabilized_plutonium_gram * (1-gallium_content)
    material_grams[isotope] = plutonium_isotope_grams
    plutonium_isotope_grams_str = str(np.round(plutonium_isotope_grams, 2))
    neutron_rate = number_produced_SF_neutrons(isotope, plutonium_isotope_grams)
    neutron_rates[isotope] = neutron_rate
    print(f"SF neutron rate, {isotope}".ljust(24) + f" ({plutonium_isotope_grams_str} g):".ljust(26), str(np.round(neutron_rate,2)).ljust(10), "n/s")

total_strength = sum(neutron_rates.values())
print("Total neutron production:".ljust(50), str(np.round(total_strength, 2)).ljust(10), "n/s")

# 2.1) Use OpenMC to simulate the neutron transport for the induced fission rates

WgPu = openmc.Material(name="WGPu")
WgPu.add_element("Ga", gallium_content, "wo")
WgPu.add_nuclide("Pu238", wgpu_3013_fractions[0][1] * (1-gallium_content), "wo")
WgPu.add_nuclide("Pu239", wgpu_3013_fractions[1][1] * (1-gallium_content), "wo")
WgPu.add_nuclide("Pu240", wgpu_3013_fractions[2][1] * (1-gallium_content), "wo")
WgPu.add_nuclide("Pu241", wgpu_3013_fractions[3][1] * (1-gallium_content), "wo")
WgPu.add_nuclide("Pu242", wgpu_3013_fractions[4][1] * (1-gallium_content), "wo")
WgPu.set_density("g/cm3", 15.76)

mats = openmc.Materials([WgPu])
mats.cross_sections = crosssection_path + "/cross_sections.xml"
mats.export_to_xml()

# determine the radius of the sphere the plutonium takes
plutonium_volume = gallium_stabilized_plutonium_gram / WgPu.density
radius_plutonium_sphere = (3 * plutonium_volume / (4 * np.pi))**(1/3)

print("\nRadius of d-phase pu sphere:".ljust(51), str(np.round(radius_plutonium_sphere, 2)).ljust(10), "cm")

s = openmc.Sphere(r=radius_plutonium_sphere, boundary_type='vacuum')
c = openmc.Cell(fill=WgPu, region=-s)
universe = openmc.Universe(cells=[c])
geometry = openmc.Geometry(universe)
geometry.export_to_xml()

settings = openmc.Settings()
settings.source = []

energies_MeV = np.arange(0, 10, 0.001)[1:]
energies_eV = 1e6 * energies_MeV

cos_theta = openmc.stats.Uniform(-1, 1)
phi = openmc.stats.Uniform(0., 2*np.pi)
r = openmc.stats.PowerLaw(0, radius_plutonium_sphere, 2)

for isotope in wgpu_3013_fractions:
    iso_source = openmc.IndependentSource()
    _, b = SF_watt_constants[isotope[0]]
    # the energy spectrum of the neutrons is now closer to the one of fast fission neutrons
    a = fast_fission_watt_constant_a[isotope[0]]
    intensities = [watt_distribution(a, b, i) for i in energies_MeV]
    iso_source.energy = openmc.stats.Discrete(energies_eV, intensities)
    iso_source.particle = 'neutron'
    iso_source.space = openmc.stats.SphericalIndependent(r=r, cos_theta=cos_theta, phi=phi)
    iso_source.angle = openmc.stats.Isotropic()
    iso_source.strength = neutron_rates[isotope[0]]
    settings.source.append(iso_source)

settings.particles = int(1e5)
batches = 100
settings.batches = batches
settings.run_mode = 'fixed source'
settings.statepoint_file = "statepoint"
settings.verbosity = 1
settings.export_to_xml()

tallies = openmc.Tallies()
fission_tally = openmc.Tally(name="fission")
fission_tally.scores = ['fission']
fission_tally.nuclides = ['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']
tallies.append(fission_tally)

nu_tally = openmc.Tally(name="nu-fission")
nu_tally.scores = ['nu-fission']
nu_tally.nuclides = ['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']
tallies.append(nu_tally)

tallies.export_to_xml()

openmc.run(openmc_exec=openmc_path, threads=4)

# 2.2) Analyze the OpenMC output to determine the induced fission rates.
sp = openmc.StatePoint("statepoint." + str(int(batches)) + ".h5")
tally = sp.get_tally(name="fission")
fissions = tally.get_values(scores=["fission"], nuclides=['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']).flatten()

print("\nTotal number of fissions:".ljust(51), str(np.round(sum(fissions),3)).ljust(9), " fissions")
print("Average fissions per neutron:".ljust(50), str(np.round(sum(fissions)/total_strength,3)).ljust(9), " fissions/neutron")

tally = sp.get_tally(name="nu-fission")
nu_fissions = tally.get_values(scores=["nu-fission"], nuclides=['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']).flatten()

print("The average multiplicity of the fissions:".ljust(50), np.round(sum(nu_fissions)/sum(fissions),3))
print("Average neutrons produced per neutron:".ljust(50), np.round(sum(nu_fissions)/total_strength,3))

print("\nNumber of induced fissions per second per atom:")
fissions_per_second_per_atom = {}
for nuclide, fission in zip(tally.nuclides, fissions):
    number_of_isotopes = (material_grams[f"{nuclide}m0"] / (atomic_weights[f"{nuclide}m0"]/1e6) * Avogadro)
    fissions_per_second_per_atom[nuclide] = fission / number_of_isotopes
    print(f"{nuclide}m0".ljust(12), str(fission / number_of_isotopes).ljust(32), "fissions/s/atom")

# 3) Use decaypy to determine the decay products (including spontaneous and induced fission)
wgpu_sphere = pd.DataFrame({'A': [238, 239, 240, 241, 242],                                      
                            'Z': [94, 94, 94, 94, 94],                                           
                            'Elevel': [0.0, 0.0, 0.0, 0.0, 0.0],                                 
                            'Amount (gram)': [material_grams[nuclide] for nuclide, _ in wgpu_3013_fractions],
                            'Decay_time (sec)': 5*[25*365*24*3600]})                             

# the second element in the tuple defines the type of fission: "SF": spontaneous fission, "FF": fast fission, "TF": thermal fission, "DTF": 14.1 MeV fission 
fissions_per_second_per_atom = {(238, 94, 0.0): (fissions_per_second_per_atom["Pu238"], "FF"),
                                (239, 94, 0.0): (fissions_per_second_per_atom["Pu239"], "FF"),
                                (240, 94, 0.0): (fissions_per_second_per_atom["Pu240"], "FF"),
                                (241, 94, 0.0): (fissions_per_second_per_atom["Pu241"], "FF"),
                                (242, 94, 0.0): (fissions_per_second_per_atom["Pu242"], "FF")}

decay = Decay(include_sf=True,
              fission_yield_reduction=False,
              induced_fission_rates=fissions_per_second_per_atom)

# takes about 10-15 minutes
isotopic_mixture = decay.decay_isotopic_mixture(wgpu_sphere)
isotopic_mixture.to_csv('composition_decayed_wgpu.csv', index=False)


# 4) Use decaypy to determine the created emissions from the determined isotopic mixture
# takes about 5 minutes
isotopic_mixture = pd.read_csv('composition_decayed_wgpu.csv')
gamma_emissions = decay.return_gamma_emissions_isotopic_mixture(isotopic_mixture=isotopic_mixture)
gamma_emissions.to_csv('gamma_emissions_decayed_wgpu.csv', index=False)

# 5) Use the OpenMCconnector to create an OpenMC material from the decayed composition 
# set the crosssection_xml_path to your custom path
crosssection_xml_path = "/home/cpf/all_openmc_xsections/official_libraries/endfb80_hdf5/cross_sections.xml"
openmc_material, not_considered_isotopes = inventory_to_openmc_material(isotopic_mixture, crosssection_path=crosssection_xml_path)

# the openmc_material is a string, that can be either stored or directly applied to an OpenMC material object via the apply_material_lines function
material = apply_material_lines(openmc_material)
material.set_density('g/cm3', 15.76)
material.name = "Decayed WGPu"
print(material)
