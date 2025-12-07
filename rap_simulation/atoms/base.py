"""Base class for atomic species."""

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Atom(ABC):
    """
    Base class for atomic species used in quantum simulations.
    
    Attributes:
        name: Human-readable name of the atom/isotope.
        transition_frequency: The frequency of the relevant atomic transition (Hz).
    """
    name: str
    transition_frequency: float  # Hz
    
    @abstractmethod
    def zeeman_shift(self, B: float) -> float:
        """
        Calculate the Zeeman shift for a given magnetic field.
        
        Args:
            B: Magnetic field strength in Tesla.
            
        Returns:
            The frequency shift in Hz.
        """
        pass
    
    @property
    def omega_0(self) -> float:
        """Angular frequency of the transition (rad/s)."""
        import numpy as np
        return 2 * np.pi * self.transition_frequency

