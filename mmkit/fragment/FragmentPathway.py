from typing import List, Tuple, Union, Sequence, Optional, Iterator
from dataclasses import dataclass, field
import json
import re

from .CleavagePattern import CleavagePattern
from .FragmentTree import FragmentTree, FragmentEdge, FragmentNode
from .HydrogenRearrangedFragmentTree import HydrogenRearrangedFragmentTree
from ..chem.Formula import Formula
from ..chem.formula_utils import assign_formulas_to_peaks
from ..mass.Tolerance import MassTolerance
from ..mass.Adduct import Adduct, split_adduct_components
from ..chem.Compound import Compound


@dataclass(frozen=True)
class FragmentPathwayGroup:
    """
    A container for multiple FragmentPathway objects.
    """
    pathways: Tuple['FragmentPathway', ...]

    def __post_init__(self) -> None:
        assert isinstance(self.pathways, tuple), "pathways must be a tuple"
        for p in self.pathways:
            assert isinstance(p, FragmentPathway), f"All items must be FragmentPathway, got {type(p)}"
    
    @staticmethod
    def from_list(
        pathways: Sequence['FragmentPathway'],
    ) -> "FragmentPathwayGroup":
        """Create a group from a list (or any sequence) of FragmentPathway."""
        return FragmentPathwayGroup(pathways=tuple(pathways))

    def __len__(self) -> int:
        return len(self.pathways)

    def __iter__(self) -> Iterator['FragmentPathway']:
        return iter(self.pathways)

    def __getitem__(self, idx: int) -> 'FragmentPathway':
        return self.pathways[idx]

    def __repr__(self) -> str:
        return f"FragmentPathwayGroup(pathways={len(self)})"
    
    def __str__(self):
        pathway_strs = []
        for fp in self.pathways:
            pathway_strs.append(f'{str(fp)}|"{str(fp.formula)}"|"{str(fp.adduct)}"')
        return '[' + ", ".join(pathway_strs) + ']'

    @staticmethod
    def parse(pathway_string: str) -> 'FragmentPathwayGroup':
        # Parse a serialized fragment pathway string into AdductedFragmentPathway objects.
        if not isinstance(pathway_string, str):
            raise ValueError("pathway_string must be a string")
        if pathway_string.strip() == "":
            return FragmentPathwayGroup.from_list([])
        # --------------------------------------------------------------
        # Helper 1: Replace quoted substrings ("...") with tokens (<MASK_0>, ...)
        # to prevent regex from being confused by commas or brackets inside quotes.
        # --------------------------------------------------------------
        def mask_quoted_text(text: str):
            quoted_texts = re.findall(r'"([^"]*)"', text)
            masked_text = text
            token_map = {}
            counter = 0

            for quoted in quoted_texts:
                token = f"<MASK_{counter}>"
                # Ensure the token name is unique in the text
                while token in masked_text:
                    counter += 1
                    token = f"<MASK_{counter}>"
                token_map[token] = quoted
                masked_text = re.sub(r'"[^"]*"', token, masked_text, count=1)
                counter += 1

            return masked_text, token_map

        # --------------------------------------------------------------
        # Helper 2: Restore masked tokens to their original quoted values
        # --------------------------------------------------------------
        def unmask_quoted_text(text: str, token_map: dict):
            for token, quoted in token_map.items():
                text = text.replace(token, f'"{quoted}"')
            return text

        # --------------------------------------------------------------
        # Helper 3: Check if only whitespace exists between two indices
        # --------------------------------------------------------------
        def is_blank_between(text: str, left_idx: int, right_idx: int) -> bool:
            return text[left_idx + 1:right_idx].strip() == ""

        # --------------------------------------------------------------
        # Step 1: Mask quotes for safe parsing
        # --------------------------------------------------------------
        masked_text, token_map = mask_quoted_text(pathway_string)

        # --------------------------------------------------------------
        # Step 2: Detect fragment edge groups using regex
        # Example match: ([...], [...], (CleavagePattern;...))
        # --------------------------------------------------------------
        pattern = re.compile(
            r'\(\s*\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*,\s*\(CleavagePattern;.*?\)\s*\)'
        )
        matches = list(pattern.finditer(masked_text))

        edge_groups = []     # Stores dicts: {steps, start, end}
        current_edges = []   # Temporarily store FragmentSteps within one group
        group_start_index = -1

        # --------------------------------------------------------------
        # Step 3: Group all fragment edges
        # --------------------------------------------------------------
        for i, match in enumerate(matches):
            start, end = match.span()
            restored_section = unmask_quoted_text(masked_text[start:end], token_map)
            current_edges.append(FragmentStep.parse(restored_section))

            if group_start_index == -1:
                # Find nearest "[" or "(" before this group
                nearest_square = masked_text.rfind("[", 0, start)
                nearest_round = masked_text.rfind("(", 0, start)
                nearest_bracket = max(nearest_square, nearest_round)

                # Error: unexpected characters between brackets
                if not is_blank_between(masked_text, nearest_bracket, start):
                    # Example: "(extra [ ... ])" → extra is invalid
                    raise ValueError("Unexpected content before fragment pathway edge.")
                group_start_index = nearest_bracket

            # Check if next match is separated by commas → same group continues
            if i + 1 < len(matches):
                next_start = matches[i + 1].start()
                in_between = masked_text[end:next_start].strip()
                if re.fullmatch(r",+", in_between):
                    continue  # still same group

            # Find right bracket closing this group
            close_square = masked_text.find("]", end)
            close_round = masked_text.find(")", end)
            if close_square == -1 and close_round == -1:
                # No closing bracket found → string broken
                raise ValueError("Malformed fragment pathway string (missing closing bracket).")

            right_bracket_idx = (
                close_round if close_square == -1
                else close_square if close_round == -1
                else min(close_square, close_round)
            )

            # Save one complete group
            edge_groups.append({
                'steps': FragmentPathwayEdge(current_edges),
                'start': group_start_index,
                'end': right_bracket_idx + 1,
            })
            current_edges = []
            group_start_index = -1

        # --------------------------------------------------------------
        # Step 4: Parse the main pathway structure and adducts
        # --------------------------------------------------------------
        parsed_pathways: List[FragmentPathway] = []
        text_index = 0
        next_edge_group_idx = 0
        depth_level = 0

        while text_index < len(masked_text):
            char = masked_text[text_index]
            matching_bracket = {"(": ")", ")": "(", "[": "]", "]": "["}.get(char, "")

            # Update nesting depth
            if char in "([":  
                depth_level += 1
            elif char in ")]":  
                depth_level -= 1

            # Only parse when we are inside a 2-level nested block
            if depth_level == 2:
                parsed_items = []
                is_first_item = True
                text_index += 1

                # ------------------------------------------
                # Parse all items within this fragment pathway
                # ------------------------------------------
                while True:
                    next_comma = masked_text.find(",", text_index)
                    next_closing = masked_text.find(matching_bracket, text_index)

                    if not is_first_item and next_comma == -1 and next_closing == -1:
                        # Neither comma nor closing → broken structure
                        raise ValueError("Malformed fragment pathway (missing comma or closing bracket).")

                    choose_comma = (
                        is_first_item or
                        (next_comma != -1 and (next_closing == -1 or next_comma < next_closing))
                    )

                    if choose_comma:
                        # Ensure there is no unexpected text between commas
                        if not is_first_item and not is_blank_between(masked_text, text_index, next_comma):
                            # Example: [...], extra text, [...] → invalid
                            raise ValueError("Unexpected content between fragment edges.")
                        start_idx = text_index if is_first_item else next_comma + 1
                        end_idx = -1
                        if not is_first_item:
                            text_index = next_comma + 1
                        is_first_item = False

                        # Check if next edge group fits here
                        if next_edge_group_idx < len(edge_groups):
                            group_start = edge_groups[next_edge_group_idx]['start']
                            if is_blank_between(masked_text, start_idx, group_start):
                                end_idx = edge_groups[next_edge_group_idx]['end']
                                parsed_items.append(edge_groups[next_edge_group_idx]['steps'])
                                next_edge_group_idx += 1

                        # If not part of an edge group → normal text fragment
                        if end_idx == -1:
                            next_comma = masked_text.find(",", text_index)
                            next_closing = masked_text.find(matching_bracket, text_index)
                            if next_comma == -1 and next_closing == -1:
                                raise ValueError("Malformed pathway (no comma or closing bracket).")

                            end_idx = (
                                next_comma if (next_comma != -1 and
                                               (next_closing == -1 or next_comma < next_closing))
                                else next_closing
                            )
                            raw_fragment = masked_text[start_idx:end_idx]
                            is_precursor = False
                            if raw_fragment.strip().startswith('p'):
                                # Precursor node
                                raw_fragment = raw_fragment.strip()[1:].strip()
                                is_precursor = True
                            restored_smiles = unmask_quoted_text(raw_fragment, token_map).strip().replace('"', '')
                            parsed_items.append(FragmentPathwayNode(restored_smiles, is_precursor=is_precursor))
                        text_index = end_idx

                    elif next_closing != -1 and (next_comma == -1 or next_closing < next_comma):
                        # Closing bracket reached → end of this section
                        text_index = next_closing + 1
                        break
                    else:
                        raise ValueError("Malformed fragment pathway (ambiguous comma/closing).")

                # ------------------------------------------
                # Parse adduct and formula part after "|"
                # ------------------------------------------
                pipe_pos = masked_text.find("|", text_index)
                if pipe_pos == -1:
                    # Missing adduct separator
                    raise ValueError("Malformed fragment pathway (missing '|').")

                between_path_and_adduct = masked_text[text_index:pipe_pos].strip()
                if between_path_and_adduct != "":
                    # Example: (...) text |[...] → text is invalid
                    raise ValueError("Unexpected content between fragment pathway and adduct.")
                text_index = pipe_pos + 1

                next_comma = masked_text.find(",", text_index)
                next_closing = masked_text.find("]", text_index)
                if next_comma == -1 and next_closing == -1:
                    # Missing adduct closing bracket
                    raise ValueError("Malformed fragment pathway (missing adduct closing).")
                elif next_comma != -1 and (next_closing == -1 or next_comma < next_closing):
                    # Comma found before closing bracket → formula follows
                    formula_and_adduct_end_idx = next_comma
                elif next_closing != -1 and (next_comma == -1 or next_closing < next_comma):
                    # Closing bracket found before comma → no formula
                    formula_and_adduct_end_idx = next_closing
                else:
                    raise ValueError("Malformed fragment pathway (ambiguous adduct ending).")

                formula_and_adduct_str = masked_text[text_index:formula_and_adduct_end_idx].strip()
                formula_str, adduct_str = formula_and_adduct_str.split("|", maxsplit=1)

                # Parse formula
                formula_str = unmask_quoted_text(formula_str, token_map).strip().replace('"', '')
                formula = Formula.parse(formula_str)

                # Parse adduct
                adduct_str = unmask_quoted_text(adduct_str, token_map).strip().replace('"', '')
                adduct = Adduct.parse(adduct_str)

                fragment_pathway = FragmentPathway(tuple(parsed_items), adduct=adduct)
                parsed_pathways.append(fragment_pathway)

                depth_level -= 1
                text_index = formula_and_adduct_end_idx

            text_index += 1

        # --------------------------------------------------------------
        # Step 5: Return all parsed FragmentPathwayG
        # --------------------------------------------------------------
        return FragmentPathwayGroup.from_list(parsed_pathways)

@dataclass(frozen=True)
class FragmentPathway:
    path: Tuple[Union['FragmentPathwayNode', 'FragmentPathwayEdge']]
    adduct: Adduct
    _formula: Optional[Formula] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.path, tuple), "path must be a tuple"
        assert len(self.path) % 2 == 1, "FragmentPathway path must have an odd number of elements (nodes and edges alternating)."
        for i in range(len(self.path)):
            if i % 2 == 0:
                assert isinstance(self.path[i], FragmentPathwayNode), f"Expected FragmentPathwayNode at position {i}, got {type(self.path[i])}"
            else:
                assert isinstance(self.path[i], FragmentPathwayEdge), f"Expected FragmentPathwayEdge at position {i}, got {type(self.path[i])}"        
        
    def __str__(self):
        pathway_strs = [str(p) for p in self.path]
        return f"[" + ",".join(pathway_strs) + "]"

    def __repr__(self):
        return f"FragmentPathway(path={self.path})"

    def get_node(self, index: int) -> 'FragmentPathwayNode':
        node: FragmentPathwayNode = None
        if index >= 0:
            node = self.path[index * 2]
        else:
            node = self.path[len(self.path) + index * 2 + 1]
        assert isinstance(node, FragmentPathwayNode)
        return node
    
    def get_edge(self, index: int) -> 'FragmentPathwayEdge':
        edge: FragmentPathwayEdge = None
        if index >= 0:
            edge = self.path[index * 2 + 1]
        else:
            edge = self.path[len(self.path) + index * 2]
        assert isinstance(edge, FragmentPathwayEdge)
        return edge

    @property
    def smiles(self) -> str:
        return self.get_node(-1).smiles

    @property
    def formula(self) -> Formula:
        """
        Return the molecular formula of the pathway end fragment
        after applying the adduct.

        The formula is computed lazily and cached.
        """
        if self._formula is None:
            # Terminal node determines the fragment structure
            terminal_node = self.get_node(-1)
            compound = Compound.from_smiles(terminal_node.smiles)

            # Apply adduct to fragment formula
            computed_formula = self.adduct.calc_formula(compound.formula).normalized

            # Cache the computed formula (allowed via object.__setattr__ even if frozen)
            object.__setattr__(self, "_formula", computed_formula)

        return self._formula

    @staticmethod
    def build_pathways_for_node(fragment_tree: FragmentTree, node_id: int, adduct_type: Adduct, precursor_formula_without_hs: Formula, adduct_type_without_hs: Adduct) -> Tuple['FragmentPathway']:
        path = FragmentPathway._collect_path_to_root(fragment_tree, node_id)
        fragment_pathways: List[FragmentPathway] = []
        for p in path:
            tmp_path = []
            for i in range(len(p)):
                if i % 2 == 0:
                    compound = Compound.from_smiles(p[i].smiles)
                    if precursor_formula_without_hs.normalized.value == adduct_type_without_hs.calc_formula(compound.formula).normalized.value:
                        is_precursor = True
                    else:
                        is_precursor = False
                    tmp_path.append(FragmentPathwayNode(p[i].smiles, is_precursor=is_precursor))
                else:
                    fragment_edge: FragmentEdge = p[i]
                    cleavage_records = fragment_edge.fragment_step_strs
                    fragment_steps = [FragmentStep.parse(record_str) for record_str in cleavage_records]
                    fragment_pathway_edge = FragmentPathwayEdge(fragment_steps)
                    tmp_path.append(fragment_pathway_edge)
            fragment_pathways.append(FragmentPathway(tuple(tmp_path), adduct=adduct_type))
        return tuple(fragment_pathways)

    @staticmethod
    def _collect_path_to_root(tree:FragmentTree, node_id: int) -> List[List[Union[FragmentNode, FragmentEdge]]]:
        """
        Recursively collect all possible paths (as lists of str(node) and str(edge))
        from the given node up to the root.
        Returns:
            A list of paths, each path being a list of strings ordered from root → current node.
        """
        node = tree.get_node(node_id)

        # Base case: node has no parents (root)
        in_edges = tree.get_in_edges(node_id)
        if len(in_edges) == 0:
            return [[node]]

        all_paths: List[List[str]] = []
        # Explore all parent nodes
        for in_edge in in_edges:
            parent_node = tree.get_node(in_edge.source_id)

            # Recursively collect paths from this parent
            parent_paths = FragmentPathway._collect_path_to_root(tree, parent_node.id)

            # Append the current edge and node to each parent path
            for path in parent_paths:
                extended_path = path.copy()
                extended_path.append(in_edge)
                extended_path.append(node)
                all_paths.append(extended_path)

        return all_paths
    
class FragmentPathwayNode:
    def __init__(self, smiles: str, is_precursor: bool):
        assert isinstance(smiles, str), "smiles must be a string"
        assert isinstance(is_precursor, bool), "is_precursor must be a boolean"
        self.smiles = smiles
        self.is_precursor = is_precursor

    def __repr__(self):
        if self.is_precursor:
            return f"FragmentPathwayNode(precursor_smiles={self.smiles})"
        else:
            return f"FragmentPathwayNode(smiles={self.smiles})"

    def __str__(self):
        if self.is_precursor:
            return f'p"{self.smiles}"'
        else:
            return f'"{self.smiles}"'

class FragmentPathwayEdge:
    def __init__(self, fragment_steps: List['FragmentStep']):
        self.fragment_steps = list(set(fragment_steps))
        assert all(isinstance(step, FragmentStep) for step in self.fragment_steps), "All fragment_steps must be FragmentStep instances."

    def __repr__(self):
        return f"FragmentPathwayEdge(fragment_steps={self.fragment_steps})"
    
    def __str__(self):
        fragment_steps_strs = [str(step) for step in self.fragment_steps]
        return "[" + ",".join(fragment_steps_strs) + "]"
    
class FragmentStep:
    def __init__(self, cleavage_pattern: CleavagePattern, react_indices: Tuple[int,...], prod_indices: Tuple[int,...]):
        self.cleavage_pattern = cleavage_pattern
        self.react_indices = react_indices
        self.prod_indices = prod_indices

    def __repr__(self):
        return (f"FragmentStep(cleavage_pattern={self.cleavage_pattern}, "
                f"react_indices={self.react_indices}, "
                f"prod_indices={self.prod_indices})")

    def __str__(self):
        react_str = "[" + ",".join(map(str, self.react_indices)) + "]"
        prod_str = "[" + ",".join(map(str, self.prod_indices)) + "]"
        cleavage_str = str(self.cleavage_pattern)
        fragment_step_str = f"({react_str},{prod_str},{cleavage_str})"
        return fragment_step_str

    @classmethod
    def parse(cls, text: str) -> 'FragmentStep':
        """
        Parse a string representation of a FragmentStep and reconstruct the object.

        Args:
            text (str): String like '([1,2],[3],"CleavagePattern;...")'

        Returns:
            FragmentStep: Parsed object
        """
        # Remove surrounding parentheses
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        # --- Step 1. Extract parts ---
        # e.g. [1,2],[3],CleavagePattern...
        # We'll extract the first two [...] blocks, then the rest
        list_matches = re.findall(r'\[([^\]]*)\]', text)
        if len(list_matches) < 2:
            raise ValueError(f"Invalid FragmentStep format: {text}")

        react_indices_str, prod_indices_str = list_matches[:2]

        # Convert to integer tuples (skip empty string case)
        react_indices = tuple(int(x) for x in react_indices_str.split(",") if x.strip())
        prod_indices = tuple(int(x) for x in prod_indices_str.split(",") if x.strip())

        # --- Step 2. Extract cleavage pattern string ---
        # Find the part after the second ']' (the pattern description)
        cleavage_str_match = re.search(r'\[[^\]]*\]\s*,\s*\[[^\]]*\]\s*,(.*)', text)
        if not cleavage_str_match:
            raise ValueError(f"Cannot extract cleavage pattern part: {text}")

        cleavage_str = cleavage_str_match.group(1).strip()

        # --- Step 3. Parse CleavagePattern ---
        # You should define CleavagePattern.parse() appropriately
        cleavage_pattern = CleavagePattern.parse(cleavage_str)

        return cls(cleavage_pattern=cleavage_pattern,
                   react_indices=react_indices,
                   prod_indices=prod_indices)