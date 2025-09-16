from typing import Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions

class CleavagePattern:
    def __init__(self, name: str, smirks: str):
        """
        Cleavage pattern defined as SMIRKS.
        Args:
            name (str): Pattern name (e.g. "amide bond cleavage").
            smirks (str): SMIRKS reaction string.
        """
        self.name = name
        self.smirks = smirks
        self.rxn = rdChemReactions.ReactionFromSmarts(smirks)

        reactant_smarts = smirks.split(">>")[0]
        self.reactant_query = Chem.MolFromSmarts(reactant_smarts)

    def exists(self, mol: Chem.Mol) -> bool:
        """
        Check if the cleavage pattern exists in the molecule.
        Returns:
            bool: True if pattern is found, False otherwise.
        """
        if self.reactant_query is None:
            return False
        return mol.HasSubstructMatch(self.reactant_query)
    
    def fragment(self, mol: Chem.Mol) -> Tuple[Tuple[Chem.Mol]]:
        """Apply the SMIRKS pattern and return fragments as SMILES."""
        idx_to_map = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
        products = self.rxn.RunReactants((mol,))
        expected_n_products = self.rxn.GetNumProductTemplates()
        fragments = []

        for prods in products:
            frag_info = []
            for p in prods:
                for a in p.GetAtoms():
                    if a.GetAtomMapNum() == 0 and a.HasProp("react_atom_idx"):
                        parent_idx = int(a.GetProp("react_atom_idx"))
                        if parent_idx in idx_to_map:
                            a.SetAtomMapNum(idx_to_map[parent_idx])
                smi = Chem.MolToSmiles(p, canonical=True)
                frag_info.append(smi)
            # frag_info = tuple(sorted(frag_info))
            fragments.append(frag_info)
        assert all(len(frag_set) == expected_n_products for frag_set in fragments), "Not all fragments match the expected number of products."

        frag_groups = tuple(
            tuple(Chem.MolFromSmiles(smi) for smi in frag_set) 
            for frag_set in fragments
        )
        return frag_groups


if __name__ == "__main__":
    amide_pattern = CleavagePattern(
        name="Amide bond cleavage",
        # smirks="[C:1](=O)-[N:2]>>[C:1](=O)[*].[N:2][*]"
        smirks="[C:1](=O)-[N:2]>>[C:1](=O)[*]"
    )

    mol = Chem.MolFromSmiles("CC(=O)NC")
    print("Original:", Chem.MolToSmiles(mol))

    frags = amide_pattern.fragment(mol)
    print("Fragments:", frags)