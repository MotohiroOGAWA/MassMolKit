from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import re

class MassTolerance(ABC):
    """Base class for mass tolerance calculation (Da or ppm)."""
    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    @property
    @abstractmethod
    def unit(self) -> str:
        """Return the unit name (e.g., 'Da' or 'ppm')."""
        pass

    @abstractmethod
    def error(self, observed: float, theoretical: float) -> float:
        """Compute the signed mass error (Da or ppm)."""
        pass

    @abstractmethod
    def within(self, observed: float, theoretical: float) -> bool:
        """Check if observed value is within tolerance from theoretical."""
        pass

    @abstractmethod
    def to_da_range(self, observed: float) -> Tuple[float, float]:
        pass

    # --- Operator overloads ---
    def __add__(self, value: float) -> "MassTolerance":
        """Return new instance with increased tolerance."""
        return self.__class__(self.tolerance + value)

    def __sub__(self, value: float) -> "MassTolerance":
        """Return new instance with decreased tolerance."""
        return self.__class__(self.tolerance - value)

    def __mul__(self, value: float) -> "MassTolerance":
        """Return new instance with multiplied tolerance."""
        return self.__class__(self.tolerance * value)

    def __truediv__(self, value: float) -> "MassTolerance":
        """Return new instance with divided tolerance."""
        return self.__class__(self.tolerance / value)

    def __repr__(self):
        return f"{self.__class__.__name__}(tolerance={self.tolerance})"

class DaTolerance(MassTolerance):
    """Absolute tolerance in Daltons."""

    def __init__(self, tolerance: float):
        super().__init__(tolerance)

    @property
    def unit(self) -> str:
        return DaTolerance._unit()
    
    @staticmethod
    def _unit() -> str:
        return "Da"

    def error(self, observed: float, theoretical: float) -> float:
        return observed - theoretical

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance
    
    def to_da_range(self, theoretical: float) -> Tuple[float, float]:
        return (theoretical - self.tolerance, theoretical + self.tolerance)
    


class PpmTolerance(MassTolerance):
    """Relative tolerance in parts-per-million (ppm)."""

    def __init__(self, tolerance: float):
        super().__init__(tolerance)

    @property
    def unit(self) -> str:
        return PpmTolerance._unit()

    @staticmethod
    def _unit() -> str:
        return "ppm"

    def error(self, observed: float, theoretical: float) -> float:
        return (observed - theoretical) / theoretical * 1e6

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance
    
    def to_da_range(self, theoretical: float) -> Tuple[float, float]:
        delta = theoretical * self.tolerance / 1e6
        return (theoretical - delta, theoretical + delta)

class AnyDaPpmTolerance(MassTolerance):
    """
    Error/unit are fixed (Da or ppm) for representation.
    The 'within' check passes if either Da tolerance OR ppm tolerance is satisfied.
    """

    def __init__(
        self,
        *,
        mode: str,        # "Da" or "ppm" (representation only)
        da_tolerance: float,
        ppm_tolerance: float,
    ):
        if mode.lower() not in (DaTolerance._unit().lower(), PpmTolerance._unit().lower()):
            raise ValueError(
                f"mode must be '{DaTolerance._unit()}' or '{PpmTolerance._unit()}', got '{mode}'"
            )

        # representation tolerance (for .tolerance and .unit)
        if mode.lower() == DaTolerance._unit().lower():
            super().__init__(da_tolerance)
            self.mode = DaTolerance._unit()
        else:
            super().__init__(ppm_tolerance)
            self.mode = PpmTolerance._unit()

        # the actual checkers
        self._da_within = DaTolerance(da_tolerance)
        self._ppm_within = PpmTolerance(ppm_tolerance)

    # -------------------------------------------------
    # Fixed representation
    # -------------------------------------------------
    @property
    def unit(self) -> str:
        return self.mode

    def error(self, observed: float, theoretical: float) -> float:
        if self.mode == DaTolerance._unit():
            return observed - theoretical
        elif self.mode == PpmTolerance._unit():
            return (observed - theoretical) / theoretical * 1e6
        else:
            raise RuntimeError("Unreachable code reached in AnyDaPpmTolerance.error()")

    # -------------------------------------------------
    # OR "within"
    # -------------------------------------------------
    def within(self, observed: float, theoretical: float) -> bool:
        return (
            self._da_within.within(observed, theoretical)
            or self._ppm_within.within(observed, theoretical)
        )

    def to_da_range(self, theoretical: float) -> Tuple[float, float]:
        da_lo, da_hi = self._da_within.to_da_range(theoretical)
        ppm_lo, ppm_hi = self._ppm_within.to_da_range(theoretical)
        # OR condition => union range in Da
        return min(da_lo, ppm_lo), max(da_hi, ppm_hi)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', "
            f"tolerance={self.tolerance}{self.unit}, "
            f"da_within={self._da_within.tolerance}Da, "
            f"ppm_within={self._ppm_within.tolerance}ppm)"
        )

_ANY_RE = re.compile(
    r"""
    ^any:
    (?P<da>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)da,
    (?P<ppm>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)ppm
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

_FIXED_RE = re.compile(
    r"""
    ^
    (?P<val>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    (?P<unit>da|ppm)
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_mass_tolerance(
    tolerance_value: str,
    tolerance_unit: str,
) -> "MassTolerance":
    """
    Parse mass tolerance from arguments and dispatch to an appropriate class.

    Supported formats
    -----------------
    1) AnyDaPpmTolerance (OR condition):
        any:<da>da,<ppm>ppm
        e.g. any:0.01da,10ppm

    2) Fixed tolerance:
        <value><unit>
        e.g. 0.01da, 10ppm

    Unit argument
    -------------
    tolerance_unit:
        "da" or "ppm"
        - For AnyDaPpmTolerance: controls mode (error()/unit representation)
        - For Fixed tolerance: ignored because the suffix provides unit
    """
    unit_arg = tolerance_unit.lower()
    if unit_arg not in ("da", "ppm"):
        raise ValueError("tolerance_unit must be 'da' or 'ppm'")

    spec = tolerance_value.strip()

    # --- 1) AnyDaPpmTolerance: "any:<da>da,<ppm>ppm" ---
    m = _ANY_RE.match(spec)
    if m:
        da_within = float(m.group("da"))
        ppm_within = float(m.group("ppm"))

        if da_within <= 0:
            raise ValueError("da_within must be > 0")
        if ppm_within <= 0:
            raise ValueError("ppm_within must be > 0")

        mode = "Da" if unit_arg == "da" else "ppm"
        return AnyDaPpmTolerance(
            mode=mode,
            da_tolerance=da_within,
            ppm_tolerance=ppm_within,
        )

    # --- 2) Fixed tolerance: "<value><unit>" ---
    m = _FIXED_RE.match(spec)
    if m:
        val = float(m.group("val"))
        u = m.group("unit").lower()
        if val <= 0:
            raise ValueError("tolerance_value must be > 0")

        if u == "da":
            return DaTolerance(val)
        else:
            return PpmTolerance(val)

    raise ValueError(
        "Invalid tolerance format. Supported formats are:\n"
        "  - any:<da>da,<ppm>ppm                (e.g. any:0.01da,10ppm)\n"
        "  - <value><unit>                      (e.g. 0.01da or 10ppm)"
    )
