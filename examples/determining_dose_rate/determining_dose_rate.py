import numpy as np
from scipy.interpolate import interp1d
from decaypy.decay_and_emissions import Decay

# ============================================================================
# ICRP-74 Dose Conversion Factors
# International Commission on Radiological Protection.
# Conversion coefficients for use in radiological protection against external radiation.
# International Commission on Radiological Protection, 1996.
# Table A.21. (column 1: Photon Energy (MeV), column 5: H*(10)/Φ (pSv·cm²))
# ============================================================================
# the first datapoint at 0 MeV is added to allow extrapolation to 0 energy
photon_energy_MeV = np.array([
    0.0, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 
    0.100, 0.150, 0.200, 0.300, 0.400, 0.500, 0.600, 0.800, 1.0,
    1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0
])

dose_coefficient = np.array([
    0.0, 0.061, 0.83, 1.05, 0.81, 0.64, 0.55, 0.51, 0.53, 0.61,
    0.89, 1.20, 1.80, 2.38, 2.93, 3.44, 4.38, 5.20, 6.90, 8.60,
    11.1, 13.4, 15.5, 17.6, 21.6, 25.6
])

dcf_interp = interp1d(photon_energy_MeV, dose_coefficient, 
                      kind='linear', fill_value='extrapolate')

# ============================================================================
# 1) Determine gamma emissions after decay
# ============================================================================
print("=" * 70)
print("Analyzing 1 gram of Cs-137 after 30 years of decay")
print("=" * 70)

decay = Decay()
decay_time_s = 30 * 365 * 24 * 3600
decay_results = decay.decay_single_isotope(A=137, Z=55, Elevel=0.0, 
                                           amount_gram=1, 
                                           time_seconds=[decay_time_s])
gamma_emissions = decay.return_gamma_emissions_isotopic_mixture(
    isotopic_mixture=decay_results)

# ============================================================================
# 2) Calculate dose rate at 1 meter
# ============================================================================
gamma_emissions_MeV = gamma_emissions['Rad Ene.'] / 1e3
dcf_values_pSv_cm2 = dcf_interp(gamma_emissions_MeV)

distance_cm = 100
geom = 1.0 / (4.0 * np.pi * distance_cm**2)

emissions_per_second = gamma_emissions['Total emissions 1/sec']
dose_rate_Sv_per_s = np.sum(emissions_per_second * geom * dcf_values_pSv_cm2 * 1e-12)

dose_rate_Sv_per_h = dose_rate_Sv_per_s * 3600
dose_rate_mSv_per_year = dose_rate_Sv_per_s * 365.25 * 24 * 3600 * 1e3

# ============================================================================
# 3) Results
# ============================================================================
print(f"\nResults at {distance_cm} cm (1 meter) distance:")
print(f"Dose rate: {dose_rate_Sv_per_h:.3e} Sv/h")
print("=" * 70)



