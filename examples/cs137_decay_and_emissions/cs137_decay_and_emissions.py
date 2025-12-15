from decaypy.decay_and_emissions import Decay


#1) decay Cs-137 without fission for 30 years.
decay = Decay()
decay_results = decay.decay_single_isotope(A=137, Z=55, Elevel=0.0, amount_gram=1000, time_seconds=[30*365*24*3600])
print(decay_results)


#2) determine the gamma emissions of this isotopic mixture
print()
gamma_emissions = decay.return_gamma_emissions_isotopic_mixture(isotopic_mixture=decay_results)
print(gamma_emissions)

#3) decay Cs-137 without fission for multiple decay steps e.g. 10, 20, 30, 40, 50 years.
# this speeds up the code as the decay matrix is determined only once
decay = Decay()
decay_results_multiple_steps = decay.decay_single_isotope(A=137, Z=55, Elevel=0.0, amount_gram=1000, time_seconds=[i * (10*365*24*3600) for i in range(1, 6)])
print(decay_results_multiple_steps)