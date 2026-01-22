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

class SwitchingWithinTolerance(MassTolerance):
    """
    Error/unit are fixed (Da or ppm).
    The 'within' check switches between Da and ppm
    based on theoretical mass.
    """

    def __init__(
        self,
        *,
        mode: str,                 # "Da" or "ppm"
        da_within: float,          # Da tolerance used when theoretical < switch_mass
        ppm_within: float,         # ppm tolerance used when theoretical >= switch_mass
        switch_mass: float = 500.0,
    ):
        if mode.lower() not in (DaTolerance._unit().lower(), PpmTolerance._unit().lower()):
            raise ValueError(f"mode must be '{DaTolerance._unit()}' or '{PpmTolerance._unit()}', got '{mode}'")

        if mode.lower() == DaTolerance._unit().lower():
            super().__init__(da_within)
            self.mode = DaTolerance._unit()
        else:
            super().__init__(ppm_within)
            self.mode = PpmTolerance._unit()
        self.switch_mass = switch_mass

        self._da_within = DaTolerance(da_within)
        self._ppm_within = PpmTolerance(ppm_within)

    # -------------------------------------------------
    # Fixed representation
    # -------------------------------------------------
    @property
    def unit(self) -> str:
        return self.mode

    def error(self, observed: float, theoretical: float) -> float:
        if self.mode == "Da":
            return observed - theoretical
        else:  # ppm
            return (observed - theoretical) / theoretical * 1e6

    # -------------------------------------------------
    # Hybrid "within"
    # -------------------------------------------------
    def within(self, observed: float, theoretical: float) -> bool:
        if theoretical < self.switch_mass:
            return self._da_within.within(observed, theoretical)
        else:
            return self._ppm_within.within(observed, theoretical)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', "
            f"tolerance={self.tolerance}{self.unit}, "
            f"da_within={self._da_within.tolerance}Da, "
            f"ppm_within={self._ppm_within.tolerance}ppm, "
            f"switch_mass={self.switch_mass})"
        )

_SWITCH_RE = re.compile(
    r"""
    ^switch:
    (?P<da>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)da,
    (?P<ppm>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)ppm
    @
    (?P<switch_mass>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
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
    1) SwitchingWithinTolerance (strict):
        switch:<da>da,<ppm>ppm@<switch_mass>
        e.g. switch:0.01da,10ppm@500

    2) Fixed tolerance (strict):
        <value><unit>
        e.g. 0.01da, 10ppm

    Unit argument
    -------------
    tolerance_unit:
        "da" or "ppm"
        - For SwitchingWithinTolerance: controls mode (error()/unit representation)
        - For Fixed tolerance:
            If tolerance_value contains its own suffix (da/ppm), that suffix is used.
            Otherwise, this function raises (we keep strict behavior).
    """
    unit_arg = tolerance_unit.lower()
    if unit_arg not in ("da", "ppm"):
        raise ValueError("tolerance_unit must be 'da' or 'ppm'")

    spec = tolerance_value.strip()

    # --- 1) SwitchingWithinTolerance ---
    m = _SWITCH_RE.match(spec)
    if m:
        da_within = float(m.group("da"))
        ppm_within = float(m.group("ppm"))
        switch_mass = float(m.group("switch_mass"))

        if da_within <= 0:
            raise ValueError("da_within must be > 0")
        if ppm_within <= 0:
            raise ValueError("ppm_within must be > 0")
        if switch_mass <= 0:
            raise ValueError("switch_mass must be > 0")

        mode = "Da" if unit_arg == "da" else "ppm"
        return SwitchingWithinTolerance(
            mode=mode,
            da_within=da_within,
            ppm_within=ppm_within,
            switch_mass=switch_mass,
        )

    # --- 2) Fixed tolerance: "<value><unit>" only ---
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
        "  - switch:<da>da,<ppm>ppm@<switch_mass>  (e.g. switch:0.01da,10ppm@500)\n"
        "  - <value><unit>                        (e.g. 0.01da or 10ppm)"
    )