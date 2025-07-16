

labels = [
    ['H', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'He'],
    ['Li', 'Be', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'B', 'C', 'N', 'O', 'F', 'Ne'],
    ['Na', 'Mg', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
    ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
    ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
    ['Cs', 'Ba', '.', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
    ['Fr', 'Ra', '.', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', '.', '.', '.']
]


# # Old version
# colors = {
#     "Nonmetal": "#FF7A7A",
#     "Noble Gas": "#C2FFC2",
#     "Alkali Metal": "#FFC2C2",
#     "Alkaline Earth Metal": "#FFC2E0",
#     "Metalloid": "#C2FFFF",
#     "Halogen": "#C2E0FF",
#     "Metal": "#E5CC80",
#     "Transition Metal": "#E5CC80",
#     "Post-Transition Metal": "#E5CC80",
#     "Lanthanide": "#BCE5E5",
#     "Actinide": "#BCE5E5",
#     "Lanthanide Series": "#BCE5E5",
#     "Poor Metal": "#E5CC80",
#     "Unknown": "#FFFFFF"
# }


# elements = {
#     "H": {"atomic_number": 1, "symbol": "H", "name": "Hydrogen", "category": "Nonmetal"},
#     "He": {"atomic_number": 2, "symbol": "He", "name": "Helium", "category": "Noble Gas"},
#     "Li": {"atomic_number": 3, "symbol": "Li", "name": "Lithium", "category": "Alkali Metal"},
#     "Be": {"atomic_number": 4, "symbol": "Be", "name": "Beryllium", "category": "Alkaline Earth Metal"},
#     "B": {"atomic_number": 5, "symbol": "B", "name": "Boron", "category": "Metalloid"},
#     "C": {"atomic_number": 6, "symbol": "C", "name": "Carbon", "category": "Nonmetal"},
#     "N": {"atomic_number": 7, "symbol": "N", "name": "Nitrogen", "category": "Nonmetal"},
#     "O": {"atomic_number": 8, "symbol": "O", "name": "Oxygen", "category": "Nonmetal"},
#     "F": {"atomic_number": 9, "symbol": "F", "name": "Fluorine", "category": "Halogen"},
#     "Ne": {"atomic_number": 10, "symbol": "Ne", "name": "Neon", "category": "Noble Gas"},
#     "Na": {"atomic_number": 11, "symbol": "Na", "name": "Sodium", "category": "Alkali Metal"},
#     "Mg": {"atomic_number": 12, "symbol": "Mg", "name": "Magnesium", "category": "Alkaline Earth Metal"},
#     "Al": {"atomic_number": 13, "symbol": "Al", "name": "Aluminum", "category": "Post-Transition Metal"},
#     "Si": {"atomic_number": 14, "symbol": "Si", "name": "Silicon", "category": "Metalloid"},
#     "P": {"atomic_number": 15, "symbol": "P", "name": "Phosphorus", "category": "Nonmetal"},
#     "S": {"atomic_number": 16, "symbol": "S", "name": "Sulfur", "category": "Nonmetal"},
#     "Cl": {"atomic_number": 17, "symbol": "Cl", "name": "Chlorine", "category": "Halogen"},
#     "Ar": {"atomic_number": 18, "symbol": "Ar", "name": "Argon", "category": "Noble Gas"},
#     "K": {"atomic_number": 19, "symbol": "K", "name": "Potassium", "category": "Alkali Metal"},
#     "Ca": {"atomic_number": 20, "symbol": "Ca", "name": "Calcium", "category": "Alkaline Earth Metal"},
#     "Sc": {"atomic_number": 21, "symbol": "Sc", "name": "Scandium", "category": "Transition Metal"},
#     "Ti": {"atomic_number": 22, "symbol": "Ti", "name": "Titanium", "category": "Transition Metal"},
#     "V": {"atomic_number": 23, "symbol": "V", "name": "Vanadium", "category": "Transition Metal"},
#     "Cr": {"atomic_number": 24, "symbol": "Cr", "name": "Chromium", "category": "Transition Metal"},
#     "Mn": {"atomic_number": 25, "symbol": "Mn", "name": "Manganese", "category": "Transition Metal"},
#     "Fe": {"atomic_number": 26, "symbol": "Fe", "name": "Iron", "category": "Transition Metal"},
#     "Co": {"atomic_number": 27, "symbol": "Co", "name": "Cobalt", "category": "Transition Metal"},
#     "Ni": {"atomic_number": 28, "symbol": "Ni", "name": "Nickel", "category": "Transition Metal"},
#     "Cu": {"atomic_number": 29, "symbol": "Cu", "name": "Copper", "category": "Transition Metal"},
#     "Zn": {"atomic_number": 30, "symbol": "Zn", "name": "Zinc", "category": "Transition Metal"},
#     "Ga": {"atomic_number": 31, "symbol": "Ga", "name": "Gallium", "category": "Metal"},
#     "Ge": {"atomic_number": 32, "symbol": "Ge", "name": "Germanium", "category": "Metalloid"},
#     "As": {"atomic_number": 33, "symbol": "As", "name": "Arsenic", "category": "Metalloid"},
#     "Se": {"atomic_number": 34, "symbol": "Se", "name": "Selenium", "category": "Nonmetal"},
#     "Br": {"atomic_number": 35, "symbol": "Br", "name": "Bromine", "category": "Halogen"},
#     "Kr": {"atomic_number": 36, "symbol": "Kr", "name": "Krypton", "category": "Noble Gas"},
#     "Rb": {"atomic_number": 37, "symbol": "Rb", "name": "Rubidium", "category": "Alkali Metal"},
#     "Sr": {"atomic_number": 38, "symbol": "Sr", "name": "Strontium", "category": "Alkaline Earth Metal"},
#     "Y": {"atomic_number": 39, "symbol": "Y", "name": "Yttrium", "category": "Transition Metal"},
#     "Zr": {"atomic_number": 40, "symbol": "Zr", "name": "Zirconium", "category": "Transition Metal"},
#     "Nb": {"atomic_number": 41, "symbol": "Nb", "name": "Niobium", "category": "Transition Metal"},
#     "Mo": {"atomic_number": 42, "symbol": "Mo", "name": "Molybdenum", "category": "Transition Metal"},
#     "Tc": {"atomic_number": 43, "symbol": "Tc", "name": "Technetium", "category": "Transition Metal"},
#     "Ru": {"atomic_number": 44, "symbol": "Ru", "name": "Ruthenium", "category": "Transition Metal"},
#     "Rh": {"atomic_number": 45, "symbol": "Rh", "name": "Rhodium", "category": "Transition Metal"},
#     "Pd": {"atomic_number": 46, "symbol": "Pd", "name": "Palladium", "category": "Transition Metal"},
#     "Ag": {"atomic_number": 47, "symbol": "Ag", "name": "Silver", "category": "Transition Metal"},
#     "Cd": {"atomic_number": 48, "symbol": "Cd", "name": "Cadmium", "category": "Transition Metal"},
#     "In": {"atomic_number": 49, "symbol": "In", "name": "Indium", "category": "Post-Transition Metal"},
#     "Sn": {"atomic_number": 50, "symbol": "Sn", "name": "Tin", "category": "Post-Transition Metal"},
#     "Sb": {"atomic_number": 51, "symbol": "Sb", "name": "Antimony", "category": "Metalloid"},
#     "Te": {"atomic_number": 52, "symbol": "Te", "name": "Tellurium", "category": "Metalloid"},
#     "I": {"atomic_number": 53, "symbol": "I", "name": "Iodine", "category": "Halogen"},
#     "Xe": {"atomic_number": 54, "symbol": "Xe", "name": "Xenon", "category": "Noble Gas"},
#     "Cs": {"atomic_number": 55, "symbol": "Cs", "name": "Cesium", "category": "Alkali Metal"},
#     "Ba": {"atomic_number": 56, "symbol": "Ba", "name": "Barium", "category": "Alkaline Earth Metal"},
#     "La": {"atomic_number": 57, "symbol": "La", "name": "Lanthanum", "category": "Lanthanide Series"},
#     "Ce": {"atomic_number": 58, "symbol": "Ce", "name": "Cerium", "category": "Lanthanide Series"},
#     "Pr": {"atomic_number": 59, "symbol": "Pr", "name": "Praseodymium", "category": "Lanthanide Series"},
#     "Nd": {"atomic_number": 60, "symbol": "Nd", "name": "Neodymium", "category": "Lanthanide Series"},
#     "Pm": {"atomic_number": 61, "symbol": "Pm", "name": "Promethium", "category": "Lanthanide Series"},
#     "Sm": {"atomic_number": 62, "symbol": "Sm", "name": "Samarium", "category": "Lanthanide Series"},
#     "Eu": {"atomic_number": 63, "symbol": "Eu", "name": "Europium", "category": "Lanthanide Series"},
#     "Gd": {"atomic_number": 64, "symbol": "Gd", "name": "Gadolinium", "category": "Lanthanide Series"},
#     "Tb": {"atomic_number": 65, "symbol": "Tb", "name": "Terbium", "category": "Lanthanide Series"},
#     "Dy": {"atomic_number": 66, "symbol": "Dy", "name": "Dysprosium", "category": "Lanthanide Series"},
#     "Ho": {"atomic_number": 67, "symbol": "Ho", "name": "Holmium", "category": "Lanthanide"},
#     "Er": {"atomic_number": 68, "symbol": "Er", "name": "Erbium", "category": "Lanthanide"},
#     "Tm": {"atomic_number": 69, "symbol": "Tm", "name": "Thulium", "category": "Lanthanide"},
#     "Yb": {"atomic_number": 70, "symbol": "Yb", "name": "Ytterbium", "category": "Lanthanide"},
#     "Lu": {"atomic_number": 71, "symbol": "Lu", "name": "Lutetium", "category": "Lanthanide"},
#     "Hf": {"atomic_number": 72, "symbol": "Hf", "name": "Hafnium", "category": "Transition Metal"},
#     "Ta": {"atomic_number": 73, "symbol": "Ta", "name": "Tantalum", "category": "Transition Metal"},
#     "W": {"atomic_number": 74, "symbol": "W", "name": "Tungsten", "category": "Transition Metal"},
#     "Re": {"atomic_number": 75, "symbol": "Re", "name": "Rhenium", "category": "Transition Metal"},
#     "Os": {"atomic_number": 76, "symbol": "Os", "name": "Osmium", "category": "Transition Metal"},
#     "Ir": {"atomic_number": 77, "symbol": "Ir", "name": "Iridium", "category": "Transition Metal"},
#     "Pt": {"atomic_number": 78, "symbol": "Pt", "name": "Platinum", "category": "Transition Metal"},
#     "Au": {"atomic_number": 79, "symbol": "Au", "name": "Gold", "category": "Transition Metal"},
#     "Hg": {"atomic_number": 80, "symbol": "Hg", "name": "Mercury", "category": "Transition Metal"},
#     "Tl": {"atomic_number": 81, "symbol": "Tl", "name": "Thallium", "category": "Poor Metal"},
#     "Pb": {"atomic_number": 82, "symbol": "Pb", "name": "Lead", "category": "Poor Metal"},
#     "Bi": {"atomic_number": 83, "symbol": "Bi", "name": "Bismuth", "category": "Poor Metal"},
#     "Po": {"atomic_number": 84, "symbol": "Po", "name": "Polonium", "category": "Metalloid"},
#     "At": {"atomic_number": 85, "symbol": "At", "name": "Astatine", "category": "Halogen"},
#     "Rn": {"atomic_number": 86, "symbol": "Rn", "name": "Radon", "category": "Noble Gas"},
#     "Fr": {"atomic_number": 87, "symbol": "Fr", "name": "Francium", "category": "Alkali Metal"},
#     "Ra": {"atomic_number": 88, "symbol": "Ra", "name": "Radium", "category": "Alkaline Earth Metal"},
#     "Ac": {"atomic_number": 89, "symbol": "Ac", "name": "Actinium", "category": "Actinide"},
#     "Th": {"atomic_number": 90, "symbol": "Th", "name": "Thorium", "category": "Actinide"},
#     "Pa": {"atomic_number": 91, "symbol": "Pa", "name": "Protactinium", "category": "Actinide"},
#     "U": {"atomic_number": 92, "symbol": "U", "name": "Uranium", "category": "Actinide"},
#     "Np": {"atomic_number": 93, "symbol": "Np", "name": "Neptunium", "category": "Actinide"},
#     "Pu": {"atomic_number": 94, "symbol": "Pu", "name": "Plutonium", "category": "Actinide"},
#     "Am": {"atomic_number": 95, "symbol": "Am", "name": "Americium", "category": "Actinide"},
#     "Cm": {"atomic_number": 96, "symbol": "Cm", "name": "Curium", "category": "Actinide"},
#     "Bk": {"atomic_number": 97, "symbol": "Bk", "name": "Berkelium", "category": "Actinide"},
#     "Cf": {"atomic_number": 98, "symbol": "Cf", "name": "Californium", "category": "Actinide"},
#     "Es": {"atomic_number": 99, "symbol": "Es", "name": "Einsteinium", "category": "Actinide"},
#     "Fm": {"atomic_number": 100, "symbol": "Fm", "name": "Fermium", "category": "Actinide"},
#     "Md": {"atomic_number": 101, "symbol": "Md", "name": "Mendelevium", "category": "Actinide"},
#     "No": {"atomic_number": 102, "symbol": "No", "name": "Nobelium", "category": "Actinide"},
#     "Lr": {"atomic_number": 103, "symbol": "Lr", "name": "Lawrencium", "category": "Actinide"},
#     "Rf": {"atomic_number": 104, "symbol": "Rf", "name": "Rutherfordium", "category": "Transition Metal"},
#     "Db": {"atomic_number": 105, "symbol": "Db", "name": "Dubnium", "category": "Transition Metal"},
#     "Sg": {"atomic_number": 106, "symbol": "Sg", "name": "Seaborgium", "category": "Transition Metal"},
#     "Bh": {"atomic_number": 107, "symbol": "Bh", "name": "Bohrium", "category": "Transition Metal"},
#     "Hs": {"atomic_number": 108, "symbol": "Hs", "name": "Hassium", "category": "Transition Metal"},
#     "Mt": {"atomic_number": 109, "symbol": "Mt", "name": "Meitnerium", "category": "Transition Metal"}
# }



### New version

colors = {
    "Reactive Nonmetal": "#00FF00",  # Green
    "Alkali Metal": "#FFD700",  # Gold
    "Alkaline Earth Metal": "#FFFF00",  # Yellow
    "Noble Gas": "#D19FE8",  # Purple
    "Transition Metal": "#FF8080",  # Light Red
    "Post-Transition Metal": "#ADD8E6",  # Light Blue
    "Metalloid": "#00FFFF",  # Cyan
    "Unknown": "#FFFFFF",
    "Actinide": "#FF0000",  # Red
    "Lanthanide": "#FF0000",  # Red
}





elements = {
    "H": {"atomic_number": 1, "symbol": "H", "name": "Hydrogen", "category": "Reactive Nonmetal"},
    "He": {"atomic_number": 2, "symbol": "He", "name": "Helium", "category": "Noble Gas"},
    "Li": {"atomic_number": 3, "symbol": "Li", "name": "Lithium", "category": "Alkali Metal"},
    "Be": {"atomic_number": 4, "symbol": "Be", "name": "Beryllium", "category": "Alkaline Earth Metal"},
    "B": {"atomic_number": 5, "symbol": "B", "name": "Boron", "category": "Metalloid"},
    "C": {"atomic_number": 6, "symbol": "C", "name": "Carbon", "category": "Reactive Nonmetal"},
    "N": {"atomic_number": 7, "symbol": "N", "name": "Nitrogen", "category": "Reactive Nonmetal"},
    "O": {"atomic_number": 8, "symbol": "O", "name": "Oxygen", "category": "Reactive Nonmetal"},
    "F": {"atomic_number": 9, "symbol": "F", "name": "Fluorine", "category": "Reactive Nonmetal"},
    "Ne": {"atomic_number": 10, "symbol": "Ne", "name": "Neon", "category": "Noble Gas"},
    "Na": {"atomic_number": 11, "symbol": "Na", "name": "Sodium", "category": "Alkali Metal"},
    "Mg": {"atomic_number": 12, "symbol": "Mg", "name": "Magnesium", "category": "Alkaline Earth Metal"},
    "Al": {"atomic_number": 13, "symbol": "Al", "name": "Aluminium", "category": "Post-Transition Metal"},
    "Si": {"atomic_number": 14, "symbol": "Si", "name": "Silicon", "category": "Metalloid"},
    "P": {"atomic_number": 15, "symbol": "P", "name": "Phosphorus", "category": "Reactive Nonmetal"},
    "S": {"atomic_number": 16, "symbol": "S", "name": "Sulfur", "category": "Reactive Nonmetal"},
    "Cl": {"atomic_number": 17, "symbol": "Cl", "name": "Chlorine", "category": "Reactive Nonmetal"},
    "Ar": {"atomic_number": 18, "symbol": "Ar", "name": "Argon", "category": "Noble Gas"},
    "K": {"atomic_number": 19, "symbol": "K", "name": "Potassium", "category": "Alkali Metal"},
    "Ca": {"atomic_number": 20, "symbol": "Ca", "name": "Calcium", "category": "Alkaline Earth Metal"},
    "Sc": {"atomic_number": 21, "symbol": "Sc", "name": "Scandium", "category": "Transition Metal"},
    "Ti": {"atomic_number": 22, "symbol": "Ti", "name": "Titanium", "category": "Transition Metal"},
    "V": {"atomic_number": 23, "symbol": "V", "name": "Vanadium", "category": "Transition Metal"},
    "Cr": {"atomic_number": 24, "symbol": "Cr", "name": "Chromium", "category": "Transition Metal"},
    "Mn": {"atomic_number": 25, "symbol": "Mn", "name": "Manganese", "category": "Transition Metal"},
    "Fe": {"atomic_number": 26, "symbol": "Fe", "name": "Iron", "category": "Transition Metal"},
    "Co": {"atomic_number": 27, "symbol": "Co", "name": "Cobalt", "category": "Transition Metal"},
    "Ni": {"atomic_number": 28, "symbol": "Ni", "name": "Nickel", "category": "Transition Metal"},
    "Cu": {"atomic_number": 29, "symbol": "Cu", "name": "Copper", "category": "Transition Metal"},
    "Zn": {"atomic_number": 30, "symbol": "Zn", "name": "Zinc", "category": "Transition Metal"},
    "Ga": {"atomic_number": 31, "symbol": "Ga", "name": "Gallium", "category": "Post-Transition Metal"},
    "Ge": {"atomic_number": 32, "symbol": "Ge", "name": "Germanium", "category": "Metalloid"},
    "As": {"atomic_number": 33, "symbol": "As", "name": "Arsenic", "category": "Metalloid"},
    "Se": {"atomic_number": 34, "symbol": "Se", "name": "Selenium", "category": "Reactive Nonmetal"},
    "Br": {"atomic_number": 35, "symbol": "Br", "name": "Bromine", "category": "Reactive Nonmetal"},
    "Kr": {"atomic_number": 36, "symbol": "Kr", "name": "Krypton", "category": "Noble Gas"},
    "Rb": {"atomic_number": 37, "symbol": "Rb", "name": "Rubidium", "category": "Alkali Metal"},
    "Sr": {"atomic_number": 38, "symbol": "Sr", "name": "Strontium", "category": "Alkaline Earth Metal"},
    "Y": {"atomic_number": 39, "symbol": "Y", "name": "Yttrium", "category": "Transition Metal"},
    "Zr": {"atomic_number": 40, "symbol": "Zr", "name": "Zirconium", "category": "Transition Metal"},
    "Nb": {"atomic_number": 41, "symbol": "Nb", "name": "Niobium", "category": "Transition Metal"},
    "Mo": {"atomic_number": 42, "symbol": "Mo", "name": "Molybdenum", "category": "Transition Metal"},
    "Tc": {"atomic_number": 43, "symbol": "Tc", "name": "Technetium", "category": "Transition Metal"},
    "Ru": {"atomic_number": 44, "symbol": "Ru", "name": "Ruthenium", "category": "Transition Metal"},
    "Rh": {"atomic_number": 45, "symbol": "Rh", "name": "Rhodium", "category": "Transition Metal"},
    "Pd": {"atomic_number": 46, "symbol": "Pd", "name": "Palladium", "category": "Transition Metal"},
    "Ag": {"atomic_number": 47, "symbol": "Ag", "name": "Silver", "category": "Transition Metal"},
    "Cd": {"atomic_number": 48, "symbol": "Cd", "name": "Cadmium", "category": "Transition Metal"},
    "In": {"atomic_number": 49, "symbol": "In", "name": "Indium", "category": "Post-Transition Metal"},
    "Sn": {"atomic_number": 50, "symbol": "Sn", "name": "Tin", "category": "Post-Transition Metal"},
    "Sb": {"atomic_number": 51, "symbol": "Sb", "name": "Antimony", "category": "Metalloid"},
    "Te": {"atomic_number": 52, "symbol": "Te", "name": "Tellurium", "category": "Metalloid"},
    "I": {"atomic_number": 53, "symbol": "I", "name": "Iodine", "category": "Reactive Nonmetal"}, # use copilot from here
    "Xe": {"atomic_number": 54, "symbol": "Xe", "name": "Xenon", "category": "Noble Gas"},
    "Cs": {"atomic_number": 55, "symbol": "Cs", "name": "Caesium", "category": "Alkali Metal"},
    "Ba": {"atomic_number": 56, "symbol": "Ba", "name": "Barium", "category": "Alkaline Earth Metal"},
    "La": {"atomic_number": 57, "symbol": "La", "name": "Lanthanum", "category": "Lanthanide"},
    "Ce": {"atomic_number": 58, "symbol": "Ce", "name": "Cerium", "category": "Lanthanide"},
    "Pr": {"atomic_number": 59, "symbol": "Pr", "name": "Praseodymium", "category": "Lanthanide"},
    "Nd": {"atomic_number": 60, "symbol": "Nd", "name": "Neodymium", "category": "Lanthanide"},
    "Pm": {"atomic_number": 61, "symbol": "Pm", "name": "Promethium", "category": "Lanthanide"},
    "Sm": {"atomic_number": 62, "symbol": "Sm", "name": "Samarium", "category": "Lanthanide"},
    "Eu": {"atomic_number": 63, "symbol": "Eu", "name": "Europium", "category": "Lanthanide"},
    "Gd": {"atomic_number": 64, "symbol": "Gd", "name": "Gadolinium", "category": "Lanthanide"},
    "Tb": {"atomic_number": 65, "symbol": "Tb", "name": "Terbium", "category": "Lanthanide"},
    "Dy": {"atomic_number": 66, "symbol": "Dy", "name": "Dysprosium", "category": "Lanthanide"},
    "Ho": {"atomic_number": 67, "symbol": "Ho", "name": "Holmium", "category": "Lanthanide"},
    "Er": {"atomic_number": 68, "symbol": "Er", "name": "Erbium", "category": "Lanthanide"},
    "Tm": {"atomic_number": 69, "symbol": "Tm", "name": "Thulium", "category": "Lanthanide"},
    "Yb": {"atomic_number": 70, "symbol": "Yb", "name": "Ytterbium", "category": "Lanthanide"},
    "Lu": {"atomic_number": 71, "symbol": "Lu", "name": "Lutetium", "category": "Lanthanide"},
    "Hf": {"atomic_number": 72, "symbol": "Hf", "name": "Hafnium", "category": "Transition Metal"},
    "Ta": {"atomic_number": 73, "symbol": "Ta", "name": "Tantalum", "category": "Transition Metal"},
    "W": {"atomic_number": 74, "symbol": "W", "name": "Tungsten", "category": "Transition Metal"},
    "Re": {"atomic_number": 75, "symbol": "Re", "name": "Rhenium", "category": "Transition Metal"}, 
    "Os": {"atomic_number": 76, "symbol": "Os", "name": "Osmium", "category": "Transition Metal"},
    "Ir": {"atomic_number": 77, "symbol": "Ir", "name": "Iridium", "category": "Transition Metal"},
    "Pt": {"atomic_number": 78, "symbol": "Pt", "name": "Platinum", "category": "Transition Metal"},
    "Au": {"atomic_number": 79, "symbol": "Au", "name": "Gold", "category": "Transition Metal"},
    "Hg": {"atomic_number": 80, "symbol": "Hg", "name": "Mercury", "category": "Transition Metal"},
    "Tl": {"atomic_number": 81, "symbol": "Tl", "name": "Thallium", "category": "Post-Transition Metal"},
    "Pb": {"atomic_number": 82, "symbol": "Pb", "name": "Lead", "category": "Post-Transition Metal"},
    "Bi": {"atomic_number": 83, "symbol": "Bi", "name": "Bismuth", "category": "Post-Transition Metal"},
    "Po": {"atomic_number": 84, "symbol": "Po", "name": "Polonium", "category": "Metalloid"},
    "At": {"atomic_number": 85, "symbol": "At", "name": "Astatine", "category": "Reactive Nonmetal"},
    "Rn": {"atomic_number": 86, "symbol": "Rn", "name": "Radon", "category": "Noble Gas"},
    "Fr": {"atomic_number": 87, "symbol": "Fr", "name": "Francium", "category": "Alkali Metal"},
    "Ra": {"atomic_number": 88, "symbol": "Ra", "name": "Radium", "category": "Alkaline Earth Metal"},
    "Ac": {"atomic_number": 89, "symbol": "Ac", "name": "Actinium", "category": "Actinide"},
    "Th": {"atomic_number": 90, "symbol": "Th", "name": "Thorium", "category": "Actinide"},
    "Pa": {"atomic_number": 91, "symbol": "Pa", "name": "Protactinium", "category": "Actinide"},
    "U": {"atomic_number": 92, "symbol": "U", "name": "Uranium", "category": "Actinide"},
    "Np": {"atomic_number": 93, "symbol": "Np", "name": "Neptunium", "category": "Actinide"},
    "Pu": {"atomic_number": 94, "symbol": "Pu", "name": "Plutonium", "category": "Actinide"},
    "Am": {"atomic_number": 95, "symbol": "Am", "name": "Americium", "category": "Actinide"},
    "Cm": {"atomic_number": 96, "symbol": "Cm", "name": "Curium", "category": "Actinide"},
    "Bk": {"atomic_number": 97, "symbol": "Bk", "name": "Berkelium", "category": "Actinide"},
    "Cf": {"atomic_number": 98, "symbol": "Cf", "name": "Californium", "category": "Actinide"},
    "Es": {"atomic_number": 99, "symbol": "Es", "name": "Einsteinium", "category": "Actinide"},
    "Fm": {"atomic_number": 100, "symbol": "Fm", "name": "Fermium", "category": "Actinide"},
    "Md": {"atomic_number": 101, "symbol": "Md", "name": "Mendelevium", "category": "Actinide"},
    "No": {"atomic_number": 102, "symbol": "No", "name": "Nobelium", "category": "Actinide"},
    "Lr": {"atomic_number": 103, "symbol": "Lr", "name": "Lawrencium", "category": "Actinide"},
    "Rf": {"atomic_number": 104, "symbol": "Rf", "name": "Rutherfordium", "category": "Transition Metal"},
    "Db": {"atomic_number": 105, "symbol": "Db", "name": "Dubnium", "category": "Transition Metal"},
    "Sg": {"atomic_number": 106, "symbol": "Sg", "name": "Seaborgium", "category": "Transition Metal"},
    "Bh": {"atomic_number": 107, "symbol": "Bh", "name": "Bohrium", "category": "Transition Metal"},
    "Hs": {"atomic_number": 108, "symbol": "Hs", "name": "Hassium", "category": "Transition Metal"},
    "Mt": {"atomic_number": 109, "symbol": "Mt", "name": "Meitnerium", "category": "Transition Metal"},
    "Ds": {"atomic_number": 110, "symbol": "Ds", "name": "Darmstadtium", "category": "Transition Metal"},
    "Rg": {"atomic_number": 111, "symbol": "Rg", "name": "Roentgenium", "category": "Transition Metal"},
    "Cn": {"atomic_number": 112, "symbol": "Cn", "name": "Copernicium", "category": "Transition Metal"},
    "Nh": {"atomic_number": 113, "symbol": "Nh", "name": "Nihonium", "category": "Unknown"},
    "Fl": {"atomic_number": 114, "symbol": "Fl", "name": "Flerovium", "category": "Unknown"},
    "Mc": {"atomic_number": 115, "symbol": "Mc", "name": "Moscovium", "category": "Unknown"},
    "Lv": {"atomic_number": 116, "symbol": "Lv", "name": "Livermorium", "category": "Unknown"},
    "Ts": {"atomic_number": 117, "symbol": "Ts", "name": "Tennessine", "category": "Unknown"},
    "Og": {"atomic_number": 118, "symbol": "Og", "name": "Oganesson", "category": "Unknown"},
}



# import streamlit as st
# from app_store.chemistry import elements, colors, labels


# st.header("Periodic Table")

# # Initialize the session state variable if it doesn't exist
# if 'bag' not in st.session_state:
#     st.session_state.bag = {}

# if 'selected_element' not in st.session_state:
#     st.session_state.selected_element = None


# for row in labels:
#     cols = st.columns(len(row))
#     for i, symbol in enumerate(row):
#         if symbol != '.':
#             element = elements[symbol]
#             category_color = colors[element['category']]
#             button_style = f"background-color: {category_color}; color: black; border: 1px solid black; padding: 8px 16px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 2px 2px; cursor: pointer;"
#             button_html = f"<button id='{symbol}' style='{button_style}' onclick='alert(\"{symbol} selected\")'>{symbol}</button>"
#             cols[i].markdown(button_html, unsafe_allow_html=True)




# # Display the sliders and remove buttons for each element
# symbols_to_remove = []
# for symbol in list(st.session_state.bag.keys()):
#     cols = st.columns([1, 2, 30])
#     with cols[0]:
#         st.markdown(f"<h3 style='font-size: 1.5em;'>{symbol}</h3>", unsafe_allow_html=True)
#     with cols[1]:
#         if st.button(f"Remove", key=f"remove_{symbol}"):
#             symbols_to_remove.append(symbol)
#     with cols[2]:
#         quantity = st.slider(f'{symbol}', 0, 20, 1, key=f"slider_{symbol}", label_visibility='collapsed')
#         st.session_state.bag[symbol] = quantity


# # Remove the symbols from the bag
# for symbol in symbols_to_remove:
#     del st.session_state.bag[symbol]

# # Rerun the app if any elements were removed
# if symbols_to_remove:
#     st.experimental_rerun()

# # Display the updated contents of the bag
# st.header("Bag:")
# st.write(", ".join(f"{k}: {v}" for k, v in st.session_state.bag.items()))


