from abc import ABC, abstractmethod

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
        tolerance: float,          # tolerance for error/unit
        da_within: float,          # Da tolerance used when theoretical < switch_mass
        ppm_within: float,         # ppm tolerance used when theoretical >= switch_mass
        switch_mass: float = 500.0,
    ):
        super().__init__(tolerance)

        if mode.lower() not in (DaTolerance._unit().lower(), PpmTolerance._unit().lower()):
            raise ValueError(f"mode must be '{DaTolerance._unit()}' or '{PpmTolerance._unit()}', got '{mode}'")

        if mode.lower() == DaTolerance._unit().lower():
            self.mode = DaTolerance._unit()
        else:
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
