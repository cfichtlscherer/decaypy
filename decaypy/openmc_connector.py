import pandas as pd
import re 
import openmc


def extract_openmc_xs_nuclides(cross_section_check_file):
    """
    Extract nuclide names available in an OpenMC cross section library.

    Parses the contents of an OpenMC `cross_sections.xml` file and returns
    all nuclide identifiers for which cross section data are available
    (e.g. "U235", "Pu239", "Am242_m1").

    The function searches for nuclide identifiers referenced either via
    the `materials` attribute or via HDF5 file paths ending in `.h5`.

    Parameters
    ----------
    cross_section_xml_text : str
        Full text contents of an OpenMC `cross_sections.xml` file.

    Returns
    -------
    list of str
        List of nuclide names present in the cross section library.
    """
    # Regular expression to match isotope names within materials="..." and path="..."
    pattern = re.compile(r'materials="([A-Za-z0-9_]+)"|path="([A-Za-z0-9_]+)\.h5"')
    
    # Find all matches and extract isotope names
    matches = pattern.findall(cross_section_check_file)
    isotopes = [match[0] if match[0] else match[1] for match in matches]
    
    return isotopes


def inventory_to_openmc_material(decay_df, crosssection_path):
    """
    Convert an isotopic inventory DataFrame into OpenMC material definitions.

    Takes an inventory (typically produced by decay or depletion calculations)
    and generates Python code lines that can be directly inserted into an
    OpenMC input script using `Material.add_nuclide()`.

    Only nuclides for which cross section data exist in the provided
    `cross_sections.xml` are added to the main material. Nuclides without
    available cross sections are instead collected separately.

    Excited states are mapped to OpenMC metastable nuclides by appending
    the suffix `_m1` when the excitation energy (`Elevel`) is greater than zero.

    Parameters
    ----------
    decay_df : pandas.DataFrame
        Isotopic inventory containing at least the columns:
        ["Element", "A", "Elevel", "Total amount (number atoms)"].
    crosssection_path : str, optional
        Path to the OpenMC `cross_sections.xml` file used to determine which
        nuclides have available cross section data.

    Returns
    -------
    tuple of str
        (material_lines, no_data_lines)

        material_lines:
            Python code lines adding nuclides with available cross sections
            to an OpenMC material.

        no_data_lines:
            Python code lines adding nuclides without available cross sections
            to a separate placeholder material (e.g. `no_data`).
    """
    openmc_material = ""
    not_considered_isotopes = ""
    # Read the cross-section file and store valid isotopes in a set
    with open(crosssection_path, "r", encoding="utf-8") as file:
        cross_section_check_file = file.read()
    
    valid_isotopes = set(extract_openmc_xs_nuclides(cross_section_check_file))
    # Iterate over the decay_df DataFrame and add valid isotopes
    for _, row in decay_df.iterrows():
        elevel = float(row['Elevel'])
        if elevel > 0.0:
            end = "_m1"   # crosssections of OpenMC only use _m1 for excited states
        else:
            end = ""
        nuclide = f"{row['Element'].capitalize()}{int(row['A'])}{end}"
        amount = row["Total amount (number atoms)"]
        if amount > 0:
            if nuclide in valid_isotopes:
                openmc_material += f"material.add_nuclide('{nuclide}', {amount}, 'ao')\n"
            else:
                not_considered_isotopes += f"no_data.add_nuclide('{nuclide}', {amount}, 'ao')\n"

    return openmc_material, not_considered_isotopes


def apply_material_lines(material_lines_str):
    """Parse the material lines string and apply to an OpenMC material."""
    material = openmc.Material()
    
    # Pattern: material.add_nuclide('Isotope', amount, 'ao')
    pattern = r"material\.add_nuclide\('([^']+)',\s*([\d.eE+-]+),\s*'ao'\)"
    
    matches = re.findall(pattern, material_lines_str)
    
    for nuclide, amount in matches:
        material.add_nuclide(nuclide, float(amount), 'ao')
    
    return material
