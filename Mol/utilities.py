from rdkit import Chem

def to_canonical_smiles(smiles: str) -> str:
    """
    Convert a SMILES string into its canonical form using RDKit.
    
    Args:
        smiles (str): Input SMILES string.
    
    Returns:
        str: Canonical SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)

def charge_from_str(charge_str: str) -> int:
    """
    Convert a charge string to an integer.
    
    Args:
        charge_str (str): Charge string, e.g., "+", "-", "+2", "-3".
    
    Returns:
        int: Charge as an integer.
    """
    if charge_str == "":
        return 0
    elif charge_str == "+":
        return 1
    elif charge_str == "-":
        return -1
    elif charge_str.startswith("+"):
        return int(charge_str[1:])
    elif charge_str.startswith("-"):
        return int(charge_str)
    else:
        raise ValueError(f"Invalid charge string: {charge_str}")
