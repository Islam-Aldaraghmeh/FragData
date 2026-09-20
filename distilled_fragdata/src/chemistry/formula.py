"""Elemental formulas and the exact mass convention of the pinned prototype."""
import re

# Pinned ms-pred common/chem_utils.py uses RDKit most-common-isotope masses.
# These are the values in the working prototype (RDKit 2025.03.6).
ELEMENT_TO_MASS = {
    "C": 12.0, "N": 14.003074, "P": 30.97376163, "O": 15.99491462,
    "S": 31.972071, "Si": 27.97692653, "I": 126.904473,
    "H": 1.007825032, "Cl": 34.96885268, "F": 18.99840322,
    "Br": 78.9183371, "B": 11.0093054, "Se": 79.9165213,
    "Fe": 55.9349375, "Co": 58.933195, "As": 74.9215965,
    "Na": 22.98976928, "K": 38.96370668,
}
ELECTRON_MASS = 0.00054858


def formula_counts(formula):
    """Parse an elemental formula (optionally with a terminal charge suffix)."""
    if isinstance(formula, dict):
        result = {}
        for symbol, count in formula.items():
            if isinstance(count, bool) or int(count) != count or count < 0:
                raise ValueError("Invalid elemental count")
            if count:
                result[symbol] = int(count)
    elif isinstance(formula, str):
        clean = re.sub(r"[+-]\d*$", "", formula.strip())
        pieces = re.findall(r"([A-Z][a-z]?)(\d*)", clean)
        if not pieces or "".join(s + n for s, n in pieces) != clean:
            raise ValueError("Unsupported formula notation")
        result = {}
        for symbol, count in pieces:
            if symbol in result:
                raise ValueError("Repeated formula element")
            n = int(count or "1")
            if n <= 0:
                raise ValueError("Invalid elemental count")
            result[symbol] = n
    else:
        raise ValueError("Formula must be text or elemental counts")
    if any(s not in ELEMENT_TO_MASS for s in result):
        raise ValueError("Unsupported formula element")
    return dict(sorted(result.items()))


def ionic_mass(formula, charge=1):
    counts = formula_counts(formula)
    if charge == 0:
        raise ValueError("Ionic m/z requires nonzero charge")
    return (sum(ELEMENT_TO_MASS[s] * n for s, n in counts.items())
            - charge * ELECTRON_MASS) / abs(charge)


def exact_mass(formula):
    """Neutral atomic mass sum; no proton or electron adjustment is applied."""
    return sum(ELEMENT_TO_MASS[s] * n for s, n in formula_counts(formula).items())


def mass_matches(formula, charge, precursor_mz):
    """Prototype's precursor check: max(0.005 Da, 20 ppm)."""
    import math
    mz = float(precursor_mz)
    return math.isfinite(mz) and abs(ionic_mass(formula, charge) - mz) <= max(.005, mz * 20e-6)
