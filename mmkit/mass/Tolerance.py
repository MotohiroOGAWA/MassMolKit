from abc import ABC, abstractmethod

class MassTolerance(ABC):
    """Base class for mass tolerance calculation (Da or ppm)."""
    def __init__(self, tolerance: float):
        self.tolerance = tolerance

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

    def error(self, observed: float, theoretical: float) -> float:
        return observed - theoretical

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance


class PpmTolerance(MassTolerance):
    """Relative tolerance in parts-per-million (ppm)."""

    def __init__(self, tolerance: float):
        super().__init__(tolerance)

    def error(self, observed: float, theoretical: float) -> float:
        return (observed - theoretical) / theoretical * 1e6

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance
