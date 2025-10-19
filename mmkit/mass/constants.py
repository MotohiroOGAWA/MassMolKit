from enum import Enum

# Disable RDKit logging
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # Only show critical errors, suppress warnings and other messages


class IonMode(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class AdductType(Enum):
    NONE = "None"
    M_PLUS_H_POS = "[M+H]+"
    M_PLUS_NH4_POS = "[M+NH4]+"
    M_PLUS_Na_POS = "[M+Na]+"
    M_MINUS_H_NEG = "[M-H]-"



PPM = 1/1000000
DEFAULT_PPM_TOLERANCE = 100 * PPM
DEFAULT_DA_TOLERANCE = 0.01 # equiv: 100ppm of 100 m/z or 10ppm of 1000 m/z



# source: https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator