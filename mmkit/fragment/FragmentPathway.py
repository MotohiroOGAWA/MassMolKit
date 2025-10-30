from typing import List, Tuple, Union
import json
import re

from .CleavagePattern import CleavagePattern
from .AdductedFragmentTree import AdductedFragmentTree, FragmentTree, FragmentNode, FragmentEdge
from ..chem.Formula import Formula
from ..chem.formula_utils import assign_formulas_to_peaks
from ..mass.Tolerance import MassTolerance
from ..mass.constants import AdductType
from ..mass.Adduct import Adduct
from ..chem.Compound import Compound

class AdductedFragmentPathway:
    def __init__(self, fragment_pathway: 'FragmentPathway', formula: Formula, adduct: Adduct):
        self.pathway = fragment_pathway
        self.formula = formula
        self.adduct = adduct

    def __str__(self):
        return f"{str(self.pathway)}|{str(self.formula)}|{str(self.adduct)}"
    
    def __repr__(self):
        return f"AdductedFragmentPathway(fragment_pathway={repr(self.pathway)}, formula={repr(self.formula)}, adduct={repr(self.adduct)})"

    @staticmethod
    def list_to_str(fragment_pathways: List['AdductedFragmentPathway']) -> str:
        pathway_strs = []
        for fragment_pathway in fragment_pathways:
            pathway_strs.append(f'{str(fragment_pathway.pathway)}|"{str(fragment_pathway.formula)}"|"{str(fragment_pathway.adduct)}"')
        return '[' + ", ".join(pathway_strs) + ']'
    
    @staticmethod
    def parse_list(pathway_string: str) -> List['AdductedFragmentPathway']:
        # Parse a serialized fragment pathway string into AdductedFragmentPathway objects.

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
        parsed_pathways = []
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
                            parsed_items.append(FragmentPathwayNode(Compound.from_smiles(restored_smiles), is_precursor=is_precursor))
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

                fragment_pathway = FragmentPathway(parsed_items)
                parsed_pathways.append(AdductedFragmentPathway(fragment_pathway, formula, adduct))

                depth_level -= 1
                text_index = formula_and_adduct_end_idx

            text_index += 1

        # --------------------------------------------------------------
        # Step 5: Return all parsed AdductedFragmentPathway objects
        # --------------------------------------------------------------
        return parsed_pathways

    @staticmethod
    def build_pathways_by_peak(
        adducted_tree: AdductedFragmentTree, 
        precursor_type: Adduct,
        peaks_mz:List[float], 
        mass_tolerance:MassTolerance,
        ) -> List[List['AdductedFragmentPathway']]:
        all_formulas_with_node_id = adducted_tree.get_all_formulas_with_node_id(precursor_type)
        formula_candidates = [f for f in all_formulas_with_node_id.keys()]
        assigned_peaks = assign_formulas_to_peaks(
            peaks_mz=peaks_mz,
            formula_candidates=formula_candidates,
            mass_tolerance=mass_tolerance,
        )
        precursor_formula = precursor_type.calc_formula(adducted_tree.compound.formula)

        adducted_fragment_pathways_by_peak: List[List[AdductedFragmentPathway]] = []
        for i, info in enumerate(assigned_peaks):
            adducted_fragment_pathways = []
            if info['n_matches'] > 0:
                for formula_str, mass_error in zip(info['matched_formulas'], info['mass_errors']):
                    formula = Formula.parse(formula_str, store_raw=False)
                    formula_with_node_id = all_formulas_with_node_id[formula]
                    for adduct_formula_pair, node_indices in formula_with_node_id.items():
                        frag_formula, adduct = adduct_formula_pair
                        for node_id in node_indices:
                            pathways = AdductedFragmentPathway.build_pathways_for_node(adducted_tree, node_id, precursor_formula, precursor_type)
                            adducted_pathways = [AdductedFragmentPathway(p, frag_formula, adduct) for p in pathways]
                            adducted_fragment_pathways.extend(adducted_pathways)
            adducted_fragment_pathways_by_peak.append(adducted_fragment_pathways)
        return adducted_fragment_pathways_by_peak

    @staticmethod
    def build_pathways_for_node(adducted_tree: AdductedFragmentTree, node_id: int, precursor_formula: Formula, precursor_type: Adduct) -> List['FragmentPathway']:
        path = AdductedFragmentPathway._collect_path_to_root(adducted_tree, node_id)
        fragment_pathways: List[FragmentPathway] = []
        for p in path:
            tmp_path = []
            for i in range(len(p)):
                if i % 2 == 0:
                    compound = Compound.from_smiles(p[i])
                    if str(precursor_formula) == str(precursor_type.calc_formula(compound.formula)):
                        is_precursor = True
                    else:
                        is_precursor = False
                    tmp_path.append(FragmentPathwayNode(Compound.from_smiles(p[i]), is_precursor=is_precursor))
                else:
                    edge_str = p[i]
                    fragment_edge = FragmentEdge.parse(edge_str)
                    cleavage_records = fragment_edge.fragment_step_strs
                    fragment_steps = [FragmentStep.parse(record_str) for record_str in cleavage_records]
                    fragment_pathway_edge = FragmentPathwayEdge(fragment_steps)
                    tmp_path.append(fragment_pathway_edge)
            fragment_pathways.append(FragmentPathway(tmp_path))
        return fragment_pathways

    @staticmethod
    def _collect_path_to_root(tree:Union[AdductedFragmentTree, FragmentTree], node_id: int) -> List[List[str]]:
        """
        Recursively collect all possible paths (as lists of str(node) and str(edge))
        from the given node up to the root.
        Returns:
            A list of paths, each path being a list of strings ordered from root → current node.
        """
        node = tree.nodes[node_id]

        # Base case: node has no parents (root)
        if not node.parent_ids:
            return [[node.smiles]]

        all_paths: List[List[str]] = []

        # Explore all parent nodes
        for parent_id in node.parent_ids:
            edge = tree.edges[(parent_id, node_id)]

            # Recursively collect paths from this parent
            parent_paths = AdductedFragmentPathway._collect_path_to_root(tree, parent_id)

            # Append the current edge and node to each parent path
            for path in parent_paths:
                extended_path = path.copy()
                extended_path.append(str(edge))
                extended_path.append(node.smiles)
                all_paths.append(extended_path)

        return all_paths


class FragmentPathway:
    def __init__(self, path: List[Union['FragmentPathwayNode', 'FragmentPathwayEdge']]):
        for i in range(len(path)):
            if i % 2 == 0:
                assert isinstance(path[i], FragmentPathwayNode), f"Expected FragmentPathwayNode at position {i}, got {type(path[i])}"
            else:
                assert isinstance(path[i], FragmentPathwayEdge), f"Expected FragmentPathwayEdge at position {i}, got {type(path[i])}"
        self.path = path

    def __str__(self):
        pathway_strs = [str(p) for p in self.path]
        return f"[" + ",".join(pathway_strs) + "]"

    def __repr__(self):
        return f"FragmentPathway(path={self.path})"

    def get_node(self, index: int) -> 'FragmentPathwayNode':
        return self.path[index * 2]
    
    def get_edge(self, index: int) -> 'FragmentPathwayEdge':
        return self.path[index * 2 + 1]
    
class FragmentPathwayNode:
    def __init__(self, compound: Compound, is_precursor: bool):
        self.compound = compound
        self.is_precursor = is_precursor

    def __repr__(self):
        if self.is_precursor:
            return f"FragmentPathwayNode(precursor_compound={self.compound})"
        else:
            return f"FragmentPathwayNode(compound={self.compound})"

    def __str__(self):
        if self.is_precursor:
            return f'p"{self.compound.smiles}"'
        else:
            return f'"{self.compound.smiles}"'

class FragmentPathwayEdge:
    def __init__(self, fragment_steps: List['FragmentStep']):
        self.fragment_steps = list(set(fragment_steps))

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