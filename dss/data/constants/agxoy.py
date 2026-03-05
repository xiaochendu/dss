# AgxOy surface constants (standalone; same values as snowyflow for compatibility).

# Map from internal index to atomic number (for MACE and energy models)
number_to_element: dict[int, int] = {
    0: 47,  # Ag
    1: 8,   # O
    2: 1,   # mask (H)
}
num_elements: int = len(number_to_element)
mask_index: int = num_elements - 1

AGXOY_REF_ELEMENT: str = "Ag"
AGXOY_REF_FORMULA: str = "Ag"

AGXOY_BULK_ENERGIES: dict[str, float] = {
    "Ag": -2.825434684753418,  # Ag bulk PBE
    "O": -4.8005975,  # O2 reference
}

AGXOY_STOICS: dict[str, int] = {
    "Ag": 1,
    "O": 0,
}

AGXOY_INDEX_TO_SYMBOL: dict[int, str] = {
    0: "Ag",
    1: "O",
    2: "H",  # mask
}
