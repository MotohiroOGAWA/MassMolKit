from collections import defaultdict
from typing import Dict, List, Tuple
from .Adduct import Adduct
from ..chem.Formula import Formula



def split_adduct_components(adduct: Adduct, reference_adducts: Tuple[Adduct]) -> Tuple[Dict[Adduct, int], Adduct]:
    """
    Split a complex Adduct into:
        (1) a mapping of supported adduct types to their counts, and
        (2) the neutral loss/gain portion as a new neutral Adduct object.
    
    Returns:
        Tuple[
            Dict[Adduct, int],   # mapping of adduct type → count
            Adduct    # neutral portion (charge 0)
        ]
    """
    h2o = Formula.parse("H2O")
    plus_neutron = Formula.parse('+n')

    # Composition counter for supported adduct types
    adduct_composition: Dict[Adduct, int] = defaultdict(int)

    # Lists to hold neutral molecules going in/out
    neutral_formulas_in: List[Formula] = []
    neutral_formulas_out: List[Formula] = []

    if adduct.charge > 0:
        # Prepare a formula-to-type lookup for positive adducts
        positive_supported_formulas = {
            str(adt.formula_shift): adt
            for adt in reference_adducts
            if adt.charge > 0
        }

        # Iterate over all adduct subformulas
        for formula, count in adduct._adduct_formulas.items():
            if count == 0:
                continue
            
            formula_str = str(formula)

            if formula_str in positive_supported_formulas:
                adduct_type = positive_supported_formulas[formula_str]
                adduct_composition[adduct_type] += count
            else:
                # Handle unsupported or neutral species
                if count > 0:
                    if formula == h2o:
                        neutral_formulas_in.extend([formula.copy() for _ in range(count)])
                    elif formula == plus_neutron:
                        neutral_formulas_in.extend([formula.copy() for _ in range(count)])
                    else:
                        raise ValueError(
                            f"Unsupported positive adduct formula: {formula} in {adduct}"
                        )
                elif count < 0:
                    neutral_formulas_out.extend([formula.copy() for _ in range(-count)])

    elif adduct.charge < 0:
        raise NotImplementedError("Negative adduct splitting not implemented yet.")
    else:
        raise ValueError(f"Adduct charge must be non-zero to split: {adduct}")

    # Create neutral portion (charge 0)
    neutral_component = Adduct(
        ion_type=adduct._ion_type,
        n_molecules=adduct._n_molecules,
        adducts_in=neutral_formulas_in,
        adducts_out=neutral_formulas_out,
        charge_offset=0
    )
    neutral_component._charge = 0

    adduct_composition = dict(sorted(adduct_composition.items(), key=lambda x: str(x[0])))

    return adduct_composition, neutral_component