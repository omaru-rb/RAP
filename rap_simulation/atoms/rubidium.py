"""Rubidium-87 atomic species."""

import scipy.constants as const
from .base import Atom


class Rubidium87(Atom):
    """
    Rubidium-87 atom with clock transition properties.
    
    The clock transition is the |F=1, mF=0⟩ → |F=2, mF=0⟩ hyperfine transition
    at approximately 6.834 GHz.
    
    Physical constants:
        - Clock transition frequency: 6.834 682 610 904 29 GHz
        - Nuclear spin: I = 3/2
        - Ground state: 5²S₁/₂
    """
    
    # Physical constants for Rb-87
    CLOCK_FREQUENCY: float = 6.834_682_610_904_29e9  # Hz (exact NIST value)
    NUCLEAR_SPIN: float = 1.5  # I = 3/2
    
    def __init__(self, B_field: float = 0.0):
        """
        Initialize Rb-87 atom.
        
        Args:
            B_field: External magnetic field in Tesla (default: 0).
        """
        super().__init__(
            name="Rb-87",
            transition_frequency=self.CLOCK_FREQUENCY
        )
        self.B_field = B_field
        
        # Physical constants
        self._mu_B = const.physical_constants['Bohr magneton'][0]
        self._g_F1 = -1/2  # g-factor for F=1
        self._g_F2 = 1/2   # g-factor for F=2
    
    def zeeman_shift(self, B: float) -> float:
        """
        Calculate the Zeeman shift for the clock transition.
        
        For the |F=1, mF=0⟩ → |F=2, mF=0⟩ transition, the first-order
        Zeeman shift vanishes. The second-order shift is:
        
            Δν = (575.15 Hz/G²) × B²
        
        Args:
            B: Magnetic field strength in Tesla.
            
        Returns:
            Frequency shift in Hz.
        """
        # Convert Tesla to Gauss (1 T = 10000 G)
        B_gauss = B * 1e4
        # Second-order Zeeman coefficient for Rb-87 clock transition
        K2 = 575.15  # Hz/G²
        return K2 * B_gauss**2
    
    def zeeman_shift_mF(self, B: float, F: int, mF: int) -> float:
        """
        Calculate first-order Zeeman shift for a specific |F, mF⟩ state.
        
        Args:
            B: Magnetic field in Tesla.
            F: Total angular momentum quantum number (1 or 2).
            mF: Magnetic quantum number.
            
        Returns:
            Frequency shift in Hz.
        """
        if F == 1:
            g_F = self._g_F1
        elif F == 2:
            g_F = self._g_F2
        else:
            raise ValueError(f"F must be 1 or 2, got {F}")
            
        if abs(mF) > F:
            raise ValueError(f"|mF| must be ≤ F, got mF={mF}, F={F}")
        
        # First-order Zeeman shift: ΔE = g_F × μ_B × mF × B
        return g_F * self._mu_B * mF * B / const.h
    
    @property
    def effective_transition_frequency(self) -> float:
        """Transition frequency including Zeeman shift from B_field."""
        return self.transition_frequency + self.zeeman_shift(self.B_field)

