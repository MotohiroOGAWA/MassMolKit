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

