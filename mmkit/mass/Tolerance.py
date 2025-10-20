from abc import ABC, abstractmethod

class MassTolerance(ABC):
    """Base class for mass tolerance calculation (Da or ppm)."""

    @abstractmethod
    def error(self, observed: float, theoretical: float) -> float:
        """Compute the signed mass error (Da or ppm)."""
        pass

    @abstractmethod
    def within(self, observed: float, theoretical: float) -> bool:
        """Check if observed value is within tolerance from theoretical."""
        pass


class DaTolerance(MassTolerance):
    """Absolute tolerance in Daltons."""

    def __init__(self, tolerance: float):
        self.tolerance = tolerance  # e.g., 0.002 Da

    def error(self, observed: float, theoretical: float) -> float:
        return observed - theoretical

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance


class PpmTolerance(MassTolerance):
    """Relative tolerance in parts-per-million (ppm)."""

    def __init__(self, tolerance: float):
        self.tolerance = tolerance  # e.g., 10 ppm

    def error(self, observed: float, theoretical: float) -> float:
        return (observed - theoretical) / theoretical * 1e6

    def within(self, observed: float, theoretical: float) -> bool:
        return abs(self.error(observed, theoretical)) <= self.tolerance
