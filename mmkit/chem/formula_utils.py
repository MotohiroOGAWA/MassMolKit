from typing import Dict, List, Tuple, Any
from itertools import product
from typing import Generator
from.Formula import Formula
from ..mass import MassTolerance


def calculate_dbe(elements: dict[str, int]) -> float:
    """
    Calculate the double bond equivalent (DBE) from an element count dictionary.
    Only common organic elements are supported (C, H, N, O, P, S, F, Cl, Br, I).

    Args:
        elements (dict): A dictionary of element counts.

    Returns:
        float: The calculated DBE.

    Raises:
        AssertionError: If unsupported elements are present.
    """
    allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Na'}
    unsupported = set(elements.keys()) - allowed_elements
    assert not unsupported, f"Unsupported elements in formula: {unsupported}"

    C = elements.get('C', 0)
    H = elements.get('H', 0)
    O = elements.get('O', 0)
    N = elements.get('N', 0)
    P = elements.get('P', 0)
    S = elements.get('S', 0)
    X = sum([elements.get(x, 0) for x in ['F', 'Cl', 'Br', 'I']])  # Halogens
    
    dbe = (2*C + N - H - X + 2) / 2.0
    return dbe

def enumerate_possible_sub_formulas(elements: dict[str, int]) -> Generator[tuple[dict[str, int], float], None, None]:
    base_elements = [(elem, count) for elem, count in elements.items() if elem != "H"]
    max_h = elements.get("H", 0)
    
    # Create all combinations from 0 to the original count for each element (excluding H)
    ranges = [range(c + 1) for _, c in base_elements]

    for counts in product(*ranges):
        # Generate element count combinations
        temp_counts = {elem: count for (elem, _), count in zip(base_elements, counts)}
        
        # Try hydrogen counts from 0 to max_h
        for h in range(max_h + 1):
            dbe = calculate_dbe(temp_counts | {"H": h})
            if dbe < 0:
                continue  # Skip if degree of unsaturation is negative

            temp_counts["H"] = h
            if sum(temp_counts.values()) == 0:
                continue
            
            yield temp_counts.copy(), dbe

def get_possible_sub_formulas(formula: Formula, hydrogen_delta: int = 0) -> Dict[str, float]:
    """
    Generate possible sub-formulas with their degree of unsaturation.
    Returns a dictionary of sub-formula strings and their DBE values.
    """
    # Add hydrogen delta to original count
    elements = formula.elements.copy()
    elements["H"] = elements.get("H", 0) + hydrogen_delta
    elements["H"] = max(elements["H"], 0)  # prevent negative H count

    res = [
        Formula(elements, charge=0)
        for elements, dbe in enumerate_possible_sub_formulas(formula.elements)
    ]

    res = sorted(res, key=lambda f: (f.exact_mass, f.plain))

    return res
    
def assign_formulas_to_peaks(
    peaks_mz: List[float],
    formula_candidates: List["Formula"],
    mass_tolerance: MassTolerance,
) -> List[Dict[str, Any]]:
    """
    Efficiently assign candidate formulas to peaks using a two-pointer approach.

    Args:
        peaks_mz (List[float]):
            List of experimental peak m/z values (unsorted).

        formula_candidates (List[Formula]):
            List of candidate molecular formulas, each with an `exact_mass` attribute.

        mass_tolerance (MassTolerance):
            Mass tolerance object (e.g., DaTolerance or PpmTolerance) defining
            the acceptable deviation for mass matching.

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries, where each dictionary corresponds to a single input peak
            (in the same order as `peaks_mz`).  
            Each dictionary contains the following keys:

            - **"mz"** (`float`):  
              The experimental m/z value of the peak.

            - **"n_matches"** (`int`):  
              The number of formula candidates whose theoretical mass falls within
              the specified mass tolerance around this peak.

            - **"matched_formulas"** (`List[str]`):  
              List of the string representations of all matching formulas.

            - **"mass_errors"** (`List[float]`):  
              List of signed mass errors between the observed peak and each matched
              formula’s theoretical mass. The unit corresponds to the tolerance type
              (`Da` for DaTolerance, `ppm` for PpmTolerance).

    Example:
        >>> peaks = [100.0, 150.0]
        >>> formulas = [Formula("C4H8O2"), Formula("C6H12O3")]
        >>> tol = PpmTolerance(10)
        >>> results = assign_formulas_to_peaks(peaks, formulas, tol)
        >>> results[0]
        {
            'mz': 100.0,
            'n_matches': 1,
            'matched_formulas': ['C4H8O2'],
            'mass_errors': [-2.3]
        }
    """

    # --- Sort both peaks and formulas by mass ---
    sorted_peaks = sorted([(mz, i) for i, mz in enumerate(peaks_mz)], key=lambda x: x[0])
    sorted_formulas = sorted([(str(f), f.exact_mass) for f in formula_candidates], key=lambda x: x[1])

    results = [{} for _ in peaks_mz]  # preserve original order
    f_idx = 0  # formula pointer
    n_formula = len(sorted_formulas)

    # --- Iterate over peaks (small → large) ---
    for mz, orig_idx in sorted_peaks:
        matches = []

        # Move pointer to the first formula mass >= mz - tolerance
        while f_idx > 0 and mass_tolerance.tolerance >= mass_tolerance.error(mz, sorted_formulas[f_idx - 1][1]):
            f_idx -= 1

        # Check all formulas in range [mz - tol, mz + tol]
        j = f_idx
        while j < n_formula and mass_tolerance.error(mz, sorted_formulas[j][1]) >= -mass_tolerance.tolerance:
            name, exact_mass = sorted_formulas[j]
            if mass_tolerance.within(mz, exact_mass):
                matches.append((name, mass_tolerance.error(mz, exact_mass)))
            j += 1

        # Store result
        results[orig_idx] = {
            "mz": mz,
            "n_matches": len(matches),
            "matched_formulas": [m[0] for m in matches],
            "mass_errors": [m[1] for m in matches],
        }

    return results