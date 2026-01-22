import os
import json
import re
from typing import List, Optional, Dict
from pathlib import Path
from .CleavagePattern import CleavagePattern


class CleavagePatternSet:
    """
    A structured collection of CleavagePattern objects.
    Provides unified access, search, and I/O utilities.
    """

    def __init__(self, name: str = "", patterns: Optional[List[CleavagePattern]] = None):
        self.name = name
        self._patterns: List[CleavagePattern] = patterns or []

    @property
    def patterns(self) -> List[CleavagePattern]:
        """Get all cleavage patterns in the set."""
        return self._patterns

    # -------------------------------------------------------------------------
    # Basic operations
    def add(self, pattern: CleavagePattern):
        """Add a new cleavage pattern."""
        self._patterns.append(pattern)

    def extend(self, patterns: List[CleavagePattern]):
        """Add multiple patterns."""
        self._patterns.extend(patterns)

    def filter(self, keyword: str) -> List[CleavagePattern]:
        """Return all patterns whose name or SMIRKS contains the keyword."""
        keyword = keyword.lower()
        return [p for p in self._patterns if keyword in p.name.lower() or keyword in p.smirks.lower()]

    def __len__(self):
        return len(self._patterns)

    def __iter__(self):
        yield from self._patterns

    def __repr__(self):
        return f"CleavagePatternSet(name='{self.name}', n_patterns={len(self._patterns)})"

    def copy(self) -> "CleavagePatternSet":
        """Create a deep copy of the Set."""
        return CleavagePatternSet(
            name=self.name,
            patterns=[p.copy() for p in self._patterns]
        )

    def to_dict(self) -> Dict:
        """Convert set to a serializable dictionary."""
        return {
            "name": self.name,
            "patterns": [p.to_dict() for p in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CleavagePatternSet":
        """Reconstruct from a dictionary."""
        patterns = [
            CleavagePattern.from_dict(d)
            for d in data.get("patterns", [])
        ]
        return cls(name=data.get("name", ""), patterns=patterns)
    
    def to_yaml(self, file_path: str):
        """Save the set to a YAML file."""
        import yaml
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, file_path: str) -> "CleavagePatternSet":
        """Load the set from a YAML file."""
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @staticmethod
    def load_default_positive() -> "CleavagePatternSet":
        """Load the default positive ion mode cleavage pattern set."""
        file_path = os.path.join(Path(__file__).parent.parent, "fragment", "cleavage_patterns", "pos", "default.yaml")
        return CleavagePatternSet.from_yaml(file_path)

    def summary(self) -> str:
        """Return a brief summary string."""
        return f"{self.name}: {len(self._patterns)} patterns"
