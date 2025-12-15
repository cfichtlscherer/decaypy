# decaypy
__decaypy__ is a Python library for accurately computing nuclear decay processes, including both spontaneous and induced fission. 
It uses data from the Evaluated Nuclear Structure Data File (ENSDF), and incorporates supplementary information from the Nuclear Wallet Cards and ENDF/B-VIII.0 fission yields.
In addition to isotopic evolution, decaypy can determine the resulting gamma emissions.

When combined with OpenMC, decaypy also supports decay simulations in subcritical systems (0 < k_eff < 1) where neutron-induced reactions are non-negligible.

### Installation
Clone the repository and install locally:
```
cd path/to/decaypy
pip install .
```

### Nuclear Data
Processed nuclear data required to run the examples is included with the repository. Optional utilities are provided to download and update external datasets.

#### ENDF fission yields
ENDF/B fission yield data can be downloaded using the data_downloader module:
```
from decaypy.data_downloader import download_endf_fission_yields
download_endf_fission_yields()
```

#### Q-Value Data (Manual Download)
NuDat3 does not provide direct download URLs for Q-value tables. These files must be downloaded manually:

1. Open **NuDat3**: https://www.nndc.bnl.gov/nudat3/
2. Select a Q-value type (e.g., **Qα**, **Qβ⁻**, **Qβ⁺**, **Qβn**, **QEC**, **QEC(p)**).
3. Click **Export → CSV**.
4. Save each CSV to:
nndc_Q_alpha.csv
nndc_Q_beta_minus.csv
nndc_Q_beta_plus.csv
nndc_Q_beta_neutron.csv
nndc_Q_electron_capture.csv
nndc_Q_electron_capture_proton.csv

#### Other Data Notes
- Additional processed data is located in `decaypy/data_processed`.
- Tools for downloading and reprocessing data are provided in `data_downloader`.
- `nuclide_names.txt` was extracted from [PyNE](https://github.com/pyne/pyne).

#### Nuclear Wallet Cards
The file `nuclear_wallet_decay_data.csv` contains decay data extracted from the Nuclear Wallet Cards (8th Edition). This file is included for reference only, as the original source is no longer publicly available. Usage and redistribution of this file may be subject to separate terms and is not covered by the project license.

### Usage
Three example scripts demonstrate the primary functionality of the library.

- __cs137_decay_and_emissions__ 
Demonstrates decay of an individual nuclide over one or more time steps and computes the resulting gamma emissions.

- __wgpu_sphere_decay_and_emissions__ 
Models the radioactive decay of a 3 kg sphere of weapons-grade plutonium and evaluates resulting radiation emissions. The system is subcritical, with both spontaneous fission and neutron-induced fission contributing to its behavior. Neutron transport and induced fission rates are determined using [OpenMC](https://github.com/openmc-dev/openmc).

- __couple_decaypy_ONIX__ 
This example illustrates a complete workflow that transforms [ONIX](https://github.com/jlanversin/ONIX) inventory output into decaypy-compatible input format through the coupling module.

### Code Structure
```
decaypy/
├── decaypy/
│ ├── decay_and_emissions.py # Core decay and emission calculations
│ ├── decay_chain_solver.py # Solves decay chains and isotopic evolution
│ ├── determine_decay_intensities.py # Decay branching and intensities
│ ├── determine_emissions.py # Gamma and radiation emission calculations
│ ├── neutron_emissions.py # Neutron emission and spontaneous fission data
│ ├── nuclide_data_helper.py # Nuclide metadata and helper utilities
│ ├── load_data.py # Data loading and parsing utilities
│ ├── data_downloader.py # External nuclear data download utilities
│ ├── openmc_connector.py # OpenMC coupling utilities
│ ├── onix_connector.py # ONIX inventory processing
│ └── data/
│  ├── data_processed/ # Preprocessed nuclear datasets
│  ├── ENDF_FY/ # ENDF/B fission yield data
│  ├── Q_values/ # Q-value CSV files from NuDat3
│  ├── nndc_txt_download/ # Raw ENSDF TXT files from NuDat3 (not required at runtime)
│  ├── nuclear_wallet_decay_data.csv # Nuclear Wallet Cards decay data
│  └── nuclide_names.txt # Nuclide name list (from PyNE)
├── examples/ # Example scripts
├── tests/ # Test suite (manual data download required)
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

#### Additional data note, NuDat3 Decay Data (Manual TXT Download)
The module relies on the ENSDF data provided by Nudat. 
The data is already provided in processed data.
To update the data the data_module relies on the TXT files that can be downloaded from the NuDat3 radition search function.
The updated NuDat3 interface is Java-based and blocks automated crawling, so `decaypy` cannot download per-nuclide decay data programmatically.

To use NuDat3 decay data, please download the TXT files manually:

1. Open: https://www.nndc.bnl.gov/nudat3/indx_dec.jsp  
2. Use the **Z filter** and request data in chunks, for example:
   - Z = 0–35, 36–44, 45–50, 51–56, 57–62, 63–65, 66–70, 71–76, 77–81, 82–86, 87–93, 94–105  
3. For each Z range:
   - click **Search**  
   - choose **Formatted file**  
   - save the result as a `.txt` file into:

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
This license excludes all nuclear data that was downloaded from external sources.
