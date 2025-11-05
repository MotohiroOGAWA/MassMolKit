from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .CleavagePattern import CleavagePattern

class FragmentResult:
    """
    Represents the result of a single cleavage pattern applied to one molecule.
    Contains the reaction SMIRKS, the original (reactant) SMILES,
    and a list of product fragment information.
    """

    def __init__(
        self,
        cleavage: CleavagePattern,
        reactant_smiles: str,
        products: Tuple['FragmentProduct', ...],
    ):
        self._cleavage = cleavage
        self._reactant_smiles = reactant_smiles
        self._products = tuple(products)

    @property
    def cleavage(self) -> CleavagePattern:
        return self._cleavage

    @property
    def reactant_smiles(self) -> str:
        """SMILES of the original molecule before cleavage."""
        return self._reactant_smiles

    @property
    def products(self) -> Tuple['FragmentProduct', ...]:
        """Tuple of product fragment information."""
        return self._products

    def __repr__(self) -> str:
        return (
            f"FragmentResult("
            f"cleavage='{self._cleavage}', "
            f"reactant='{self._reactant_smiles}', "
            f"n_products={len(self._products)})"
        )


class FragmentProduct:
    """
    Represents a single product fragment obtained from a cleavage reaction.
    Stores both the fragment SMILES and index mapping between
    reactant and product atoms.
    """

    def __init__(
        self,
        smiles: str,
        reactant_indices: Tuple[int, ...],
        product_indices: Tuple[int, ...],
    ):
        self._smiles = smiles
        self._reactant_indices = tuple(reactant_indices)
        self._product_indices = tuple(product_indices)

    @property
    def smiles(self) -> str:
        """SMILES of this fragment."""
        return self._smiles

    @property
    def reactant_indices(self) -> Tuple[int, ...]:
        """Indices of corresponding atoms in the original molecule."""
        return self._reactant_indices

    @property
    def product_indices(self) -> Tuple[int, ...]:
        """Indices of atoms in this fragment molecule."""
        return self._product_indices

    def __repr__(self) -> str:
        return (
            f"FragmentProduct("
            f"smiles='{self._smiles}', "
            f"reactant_indices={self._reactant_indices}, "
            f"product_indices={self._product_indices})"
        )
