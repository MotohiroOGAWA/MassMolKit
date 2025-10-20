from abc import ABC, abstractmethod

class MassError(ABC):
    """Base class for mass error calculation (Da or ppm)."""

    @abstractmethod
    def compute(self, observed: float, theoretical: float) -> float:
        """
        Compute the signed mass error between observed and theoretical values.
        """
        pass

    @abstractmethod
    def within(self, observed: float, theoretical: float, tolerance: float) -> bool:
        """
        Check whether the error is within the given tolerance.
        """
        pass


class DaError(MassError):
    """Mass error calculator using absolute difference in Daltons."""

    def compute(self, observed: float, theoretical: float) -> float:
        """
        Compute mass error in Daltons (Da).
        """
        return observed - theoretical

    def within(self, observed: float, theoretical: float, tolerance: float) -> bool:
        """
        Return True if |error| <= tolerance (in Da).
        """
        return abs(self.compute(observed, theoretical)) <= tolerance


class PpmError(MassError):
    """Mass error calculator using parts-per-million (ppm) difference."""

    def compute(self, observed: float, theoretical: float) -> float:
        """
        Compute relative mass error in parts-per-million (ppm).
        """
        return (observed - theoretical) / theoretical * 1e6

    def within(self, observed: float, theoretical: float, tolerance: float) -> bool:
        """
        Return True if |error| <= tolerance (in ppm).
        """
        return abs(self.compute(observed, theoretical)) <= tolerance
