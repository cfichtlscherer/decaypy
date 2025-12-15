"""
This example illustrates a complete workflow that transforms ONIX inventory output 
into decaypy-compatible input format through the coupling module.
Once transformed, decaypy can perform comprehensive decay and emission 
calculations on the material composition.
"""

from decaypy.decay_and_emissions import Decay
from decaypy.onix_connector import extract_onix_isotopic_inventory, generate_decaypy_input

# generate the isotopic inventory from ONIX output in form of a pandas DataFrame
# the volume of the material in cm^3 is needed to convert densities to number of atoms
# the ONIX density output file path must be provided
# the ONIX output step index (0-based)
number_of_existing_atoms = extract_onix_isotopic_inventory(density_output_onix="ONIX_output_example", volume=1000.0, step=2)

# decay time in seconds
decay_time_s = 30 * 365 * 24 * 3600  # 30 years
# alternative add a list of multiple decay steps
# decay_time_s = [1*365*24*3600, 5*365*24*3600, 10*365*24*3600, 30*365*24*3600]
decaypy_input = generate_decaypy_input(number_of_existing_atoms, decay_time_s)

decay = Decay()
decay_results = decay.decay_isotopic_mixture(isotopic_mixture=decaypy_input)
print(decay_results)



