import os
import json
import re
from typing import List, Optional, Dict
from pathlib import Path
from .CleavagePattern import CleavagePattern


class CleavagePatternLibrary:
    """
    A structured collection of CleavagePattern objects.
    Provides unified access, search, and I/O utilities.
    """

    def __init__(self, name: str = "", patterns: Optional[List[CleavagePattern]] = None):
        self.name = name
        self._patterns: List[CleavagePattern] = patterns or []

    @property
    def patterns(self) -> List[CleavagePattern]:
        """Get all cleavage patterns in the library."""
        return self._patterns

    # -------------------------------------------------------------------------
    # Basic operations
    def add(self, pattern: CleavagePattern):
        """Add a new cleavage pattern."""
        self._patterns.append(pattern)

    def extend(self, patterns: List[CleavagePattern]):
        """Add multiple patterns."""
        self._patterns.extend(patterns)

    # def remove(self, name: str):
    #     """Remove a pattern by its name."""
    #     self._patterns = [p for p in self._patterns if p.name != name]

    # def get(self, name: str) -> Optional[CleavagePattern]:
    #     """Retrieve a pattern by name."""
    #     return next((p for p in self._patterns if p.name == name), None)

    # -------------------------------------------------------------------------
    # Search / filter
    def filter(self, keyword: str) -> List[CleavagePattern]:
        """Return all patterns whose name or SMIRKS contains the keyword."""
        keyword = keyword.lower()
        return [p for p in self._patterns if keyword in p.name.lower() or keyword in p.smirks.lower()]

    def __len__(self):
        return len(self._patterns)

    def __iter__(self):
        yield from self._patterns

    def __repr__(self):
        return f"CleavagePatternLibrary(name='{self.name}', n_patterns={len(self._patterns)})"

    def copy(self) -> "CleavagePatternLibrary":
        """Create a deep copy of the library."""
        return CleavagePatternLibrary(
            name=self.name,
            patterns=[p.copy() for p in self._patterns]
        )

    # -------------------------------------------------------------------------
    # Serialization
    def to_dict(self) -> Dict:
        """Convert library to a serializable dictionary."""
        return {
            "name": self.name,
            "patterns": [p.to_dict() for p in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CleavagePatternLibrary":
        """Reconstruct from a dictionary."""
        from .CleavagePattern import CleavagePattern 
        patterns = [
            CleavagePattern.from_dict(d)
            for d in data.get("patterns", [])
        ]
        return cls(name=data.get("name", ""), patterns=patterns)

    def save_json(self, path: str):
        """Save the library to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "CleavagePatternLibrary":
        """Load a library from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @staticmethod
    def load_default_positive() -> "CleavagePatternLibrary":
        """Load the default positive ion mode cleavage pattern library."""
        file_path = os.path.join(Path(__file__).parent.parent, "fragment", "cleavage_patterns", "pos", "default.json")
        return CleavagePatternLibrary.load_json(file_path)

    # -------------------------------------------------------------------------
    # Utility
    def summary(self) -> str:
        """Return a brief summary string."""
        return f"{self.name}: {len(self._patterns)} patterns"
