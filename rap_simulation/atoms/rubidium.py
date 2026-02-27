"""Rubidium-87 atomic species."""

from dataclasses import dataclass
from typing import Tuple
import scipy.constants as const
from .base import Atom


@dataclass(frozen=True)
class HyperfineState:
    """Represents a hyperfine state |F, mF⟩."""
    F: int
    mF: int
    
    def __post_init__(self):
        if self.F not in (1, 2):
            raise ValueError(f"F must be 1 or 2 for Rb-87 ground state, got {self.F}")
        if abs(self.mF) > self.F:
            raise ValueError(f"|mF| must be ≤ F, got mF={self.mF}, F={self.F}")
    
    def __str__(self) -> str:
        return f"|F={self.F}, mF={self.mF}⟩"


@dataclass(frozen=True)
class Transition:
    """
    Represents a hyperfine transition between two states.
    
    Attributes:
        initial: Initial hyperfine state.
        final: Final hyperfine state.
        name: Optional name for the transition.
    """
    initial: HyperfineState
    final: HyperfineState
    name: str = ""
    
    def __post_init__(self):
        # Validate selection rules for magnetic dipole transitions
        delta_F = abs(self.final.F - self.initial.F)
        delta_mF = abs(self.final.mF - self.initial.mF)
        
        if delta_F > 1:
            raise ValueError(f"ΔF must be 0 or ±1, got {delta_F}")
        if delta_mF > 1:
            raise ValueError(f"ΔmF must be 0 or ±1, got {delta_mF}")
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {self.initial} → {self.final}"
        return f"{self.initial} → {self.final}"


class Rubidium87(Atom):
    """
    Rubidium-87 atom with hyperfine transition properties.
    
    The ground state 5²S₁/₂ has two hyperfine levels:
        - F=1 with mF = -1, 0, +1
        - F=2 with mF = -2, -1, 0, +1, +2
    
    Physical constants:
        - Hyperfine splitting: 6.834 682 610 904 29 GHz
        - Nuclear spin: I = 3/2
        - g_F(F=1) = -1/2
        - g_F(F=2) = +1/2
    
    Common transitions:
        - Clock: |F=1, mF=0⟩ → |F=2, mF=0⟩ (field-insensitive)
        - σ+: |F=1, mF=1⟩ → |F=2, mF=2⟩ or |F=2, mF=2⟩ → |F=1, mF=1⟩
        - σ-: |F=1, mF=-1⟩ → |F=2, mF=-2⟩
        - π: ΔmF = 0 transitions
    """
    
    # Physical constants for Rb-87
    HYPERFINE_SPLITTING: float = 6.834_682_610_904_29e9  # Hz (exact NIST value)
    CLOCK_FREQUENCY: float = HYPERFINE_SPLITTING  # Alias for backwards compatibility
    NUCLEAR_SPIN: float = 1.5  # I = 3/2
    
    # Common transitions
    CLOCK_TRANSITION = Transition(
        HyperfineState(1, 0), HyperfineState(2, 0), name="clock"
    )
    STRETCH_TRANSITION = Transition(
        HyperfineState(1, 1), HyperfineState(2, 2), name="stretch (mF=1<->mF=2)"
    )
    MIDDLE_TRANSITION = Transition(
        HyperfineState(1, 1), HyperfineState(2, 0), name="middle (mF=1<->mF=0)"
    )
    
    def __init__(self, B_field: float = 0.0, transition: Transition | None = None):
        """
        Initialize Rb-87 atom.
        
        Args:
            B_field: External magnetic field in Tesla (default: 0).
            transition: The hyperfine transition to use. Defaults to clock transition.
        """
        self.B_field = B_field
        self._transition = transition or self.CLOCK_TRANSITION
        
        # Physical constants
        self._mu_B = const.physical_constants['Bohr magneton'][0]
        self._g_F1 = -1/2  # g-factor for F=1
        self._g_F2 = 1/2   # g-factor for F=2
        
        # Calculate transition frequency for the selected transition
        self._transition_frequency = self.get_transition_frequency(self._transition, B_field)
        
        super().__init__(
            name=f"Rb-87 ({self._transition.name or str(self._transition)})",
        )
    
    @property
    def transition(self) -> Transition:
        """The current transition being used."""
        return self._transition
    @property
    def transition_frequency(self) -> float:
        """The frequency of the current transition."""
        return self._transition_frequency

    def set_transition(self, transition: Transition):
        """Set the transition for the atom."""
        self._transition = transition
        self._transition_frequency = self.get_transition_frequency(transition, self.B_field)
    
    def zeeman_shift_state(self, B: float, state: HyperfineState) -> float:
        """
        Calculate first-order Zeeman shift for a hyperfine state.
        
        Args:
            B: Magnetic field in Tesla.
            state: The hyperfine state.
            
        Returns:
            Frequency shift in Hz.
        """
        if state.F == 1:
            g_F = self._g_F1
        elif state.F == 2:
            g_F = self._g_F2
        else:
            raise ValueError(f"F must be 1 or 2, got {state.F}")
        
        # First-order Zeeman shift: Δν = 1.4 MHz/G × mF × B * g_F
        return 1.4e6 * state.mF * B * 1e4* g_F
    
    def zeeman_shift(self, B: float, F: int, mF: int) -> float:
        """
        Calculate first-order Zeeman shift for a specific |F, mF⟩ state.
        
        Args:
            B: Magnetic field in Tesla.
            F: Total angular momentum quantum number (1 or 2).
            mF: Magnetic quantum number.
            
        Returns:
            Frequency shift in Hz.
        """
        return self.zeeman_shift_state(B, HyperfineState(F, mF))
    
    def second_order_zeeman_shift(self, B: float) -> float:
        """
        Calculate the Zeeman shift for the clock transition (mF=0 → mF=0).
        
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
    
    def get_transition_frequency(self, transition: Transition, B: float = 0.0) -> float:
        """
        Calculate the transition frequency including Zeeman shifts.
        
        The transition frequency is:
            ν = ν_HFS + Δν_final - Δν_initial
        
        where Δν are the Zeeman shifts of each state.
        
        Args:
            transition: The hyperfine transition.
            B: Magnetic field in Tesla.
            
        Returns:
            Transition frequency in Hz.
        """

        base_freq = self.HYPERFINE_SPLITTING
        
        # Add Zeeman shifts
        shift_initial = self.zeeman_shift_state(B, transition.initial)
        shift_final = self.zeeman_shift_state(B, transition.final)
        
        # For clock transition, add second-order shift
        if transition.initial.mF == 0 and transition.final.mF == 0:
            return base_freq + self.second_order_zeeman_shift(B)
        
        return base_freq + shift_final - shift_initial
    
    def zeeman_sensitivity(self, transition: Transition | None = None) -> float:
        """
        Calculate the first-order Zeeman sensitivity dν/dB for a transition.
        
        Args:
            transition: The transition. Defaults to current transition.
            
        Returns:
            Sensitivity in Hz/Tesla.
        """
        trans = transition or self._transition
        
        # For clock transition
        if trans.initial.mF == 0 and trans.final.mF == 0:
            return 0.0  # First-order insensitive
        
        # dν/dB = (g_F_final × mF_final - g_F_initial × mF_initial) × μ_B / h
        g_i = self._g_F1 if trans.initial.F == 1 else self._g_F2
        g_f = self._g_F1 if trans.final.F == 1 else self._g_F2
        
        return (g_f * trans.final.mF - g_i * trans.initial.mF) * self._mu_B / const.h
    
    @property
    def effective_transition_frequency(self) -> float:
        """Transition frequency including Zeeman shift from B_field."""
        return self.get_transition_frequency(self._transition, self.B_field)
    
    @classmethod
    def list_transitions(cls) -> list[Transition]:
        """Return list of pre-defined common transitions."""
        return [
            cls.CLOCK_TRANSITION,
            cls.STRETCH_TRANSITION,
            cls.MIDDLE_TRANSITION,
        ]
    
    @classmethod
    def custom_transition(
        cls, 
        F_initial: int, mF_initial: int,
        F_final: int, mF_final: int,
        name: str = ""
    ) -> Transition:
        """
        Create a custom transition.
        
        Args:
            F_initial, mF_initial: Initial state quantum numbers.
            F_final, mF_final: Final state quantum numbers.
            name: Optional name for the transition.
            
        Returns:
            Transition object.
            
        Example:
            >>> trans = Rubidium87.custom_transition(2, 1, 1, 0, name="π")
        """
        return Transition(
            HyperfineState(F_initial, mF_initial),
            HyperfineState(F_final, mF_final),
            name=name
        )

