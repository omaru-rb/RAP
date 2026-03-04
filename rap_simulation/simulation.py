"""
Core simulation module for Composite Pulses (CP).

This module provides the main CompositePulse class that handles
the quantum dynamics simulation using QuTiP.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from qutip import Qobj, basis, sesolve, sigmax, sigmaz, sigmay, expect, Result

from .pulses import get_pulse
from .detuning import get_detuning
from .phase import get_phase

if TYPE_CHECKING:
    from .atoms.base import Atom


@dataclass
class SimulationParams:
    """
    Parameters for composite pulses simulation.
    
    Attributes:
        T: Total integration time (s).
        dt: Time step for output (s).
        omega: Peak Rabi frequency (rad/s).
        sweep_time: Half-duration of the adiabatic sweep (s).
        delta_span: Rate of detuning change (rad/s per second).
        span_center: Center detuning value (rad/s).
        t_center: Center time of the passage (s). Defaults to T/2.
        phase : Phase of the pulse (rad).
    """
    T: float
    dt: float
    omega: float
    phase : float
    sweep_time: float
    freq_span: float
    freq_span_center: float = 0.0
    t_center: float | None = None
    
    def __post_init__(self):
        if self.t_center is None:
            self.t_center = self.T / 2
    
    @classmethod
    def from_atom(cls, T:float,dt:float, omega:float, phase:float, sweep_time:float, freq_span:float, freq_span_center:float) -> "SimulationParams":
        """
        Create parameters
        """
        return cls(
            T = T,
            dt = dt,
            omega = omega,
            phase=phase,
            sweep_time = sweep_time,
            freq_span_center = freq_span_center,
            freq_span = freq_span,
        )
    
    @classmethod   # TODO: fix this method
    def for_rb87_clock(
        cls,
        T: float = 1e-3,
        dt: float = 1e-6,
        rabi_freq_hz: float = 10_000,
        freq_span_hz: float = 40_000,
        sweep_time: float | None = None,
    ) -> "SimulationParams":
        """
        Create parameters suitable for Rb-87 clock transition experiments.
        
        Args:
            T: Total time (s). Default: 1 ms.
            dt: Time step (s). Default: 1 μs.
            rabi_freq_hz: Rabi frequency in Hz. Default: 10 kHz.
            freq_span_hz: Total frequency span in Hz. Default: 40 kHz.
            sweep_time: Half-sweep duration (s). Default: T/4.
            
        Returns:
            SimulationParams configured for Rb-87.
        """
        if sweep_time is None:
            sweep_time = T / 4
            
        omega = 2 * np.pi * rabi_freq_hz
        delta_span = 2 * np.pi * freq_span_hz / (2 * sweep_time)
        
        return cls(
            T=T,
            dt=dt,
            omega=omega,
            sweep_time=sweep_time,
            delta_span=delta_span,
            span_center=0.0,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for QuTiP args."""
        return {
            'T': self.T,
            'dt': self.dt,
            't_center': self.t_center,
            'omega': self.omega,
            'sweep_time': self.sweep_time,
            'freq_span': self.freq_span,
            'freq_span_center': self.freq_span_center,
            'phase' : self.phase,
        }


@dataclass
class SimulationResult:
    """
    Results from a composite pulses simulation.
    
    Attributes:
        times: Array of time points (s).
        states: List of QuTiP Qobj states at each time.
        c0: Complex amplitudes of |0⟩ state.
        c1: Complex amplitudes of |1⟩ state.
        probabilities: Dict with 'p0' and 'p1' probability arrays.
        bloch_coords: Dict with 'x', 'y', 'z' Bloch sphere coordinates.
        params: The SimulationParams used.
        pulse_name: Name of pulse shape used.
        phase_name : Name of phase profile used.
        detuning_name: Name of detuning profile used.
        qutip_result: The raw QuTiP Result object.
    """
    times: np.ndarray
    states: list[Qobj]
    c0: np.ndarray
    c1: np.ndarray
    probabilities: dict[str, np.ndarray]
    bloch_coords: dict[str, np.ndarray]
    params: SimulationParams
    pulse_name: str
    phase_name: str
    detuning_name: str
    qutip_result: Result
    
    @property
    def p0(self) -> np.ndarray:
        """Probability of |0⟩ state."""
        return self.probabilities['p0']
    
    @property
    def p1(self) -> np.ndarray:
        """Probability of |1⟩ state."""
        return self.probabilities['p1']
    
    @property
    def final_p1(self) -> float:
        """Final transfer probability to |1⟩."""
        return float(self.probabilities['p1'][-1])
    
    @property
    def max_p1(self) -> float:
        """Maximum probability achieved in |1⟩."""
        return float(np.max(self.probabilities['p1']))
    
    def expectation_values(self, operators: list[Qobj] | None = None) -> dict[str, np.ndarray]:
        """
        Calculate expectation values for given operators.
        
        Args:
            operators: List of QuTiP operators. Defaults to [σx, σy, σz].
            
        Returns:
            Dict mapping operator names to expectation value arrays.
        """
        if operators is None:
            operators = [sigmax(), sigmaz()]
            names = ['sigma_x', 'sigma_z']
        else:
            names = [f'op_{i}' for i in range(len(operators))]
        
        expects = expect(operators, self.states)
        return dict(zip(names, expects))


class CompositePulse:
    """
    Simulator for composite pulses acting on a two-level system.
    
    This class sets up and solves the time-dependent Schrödinger equation
    for a two-level system driven by a composite pulse.
    
    The Hamiltonian is:
        H(t) = (Ω(t)cos(φ)/2) σx + (Ω(t)sin(φ)/2) σy + Δ(t) σz
    
    where Ω(t) is the Rabi frequency, φ is the phase and Δ(t) is the detuning.
    
    Example:
        >>> from rap_simulation import Rubidium87, RapidAdiabaticPassage, SimulationParams
        >>> 
        >>> atom = Rubidium87()
        >>> params = SimulationParams.for_rb87_clock(T=1e-3)
        >>> 
        >>> rap = RapidAdiabaticPassage(atom, params)
        >>> result = rap.run(pulse="hyper_secant", detuning="linear")
        >>> 
        >>> print(f"Transfer efficiency: {result.final_p1:.4f}")
    """
    
    def __init__(self, atom: "Atom", params: SimulationParams):
        """
        Initialize the CP simulator.
        
        Args:
            atom: The atomic species to simulate.
            params: Simulation parameters.
        """
        self.atom = atom
        self.params = params
        
        # Basis states
        self._psi0 = basis(2, 0)  # |0⟩
        self._psi1 = basis(2, 1)  # |1⟩
    
    def _build_hamiltonian(
        self,
        pulse_name: str,
        detuning_name: str,
        phase_name: str,
        pulse_kwargs: dict | None = None,
        detuning_kwargs: dict | None = None,
        phase_kwargs: dict | None = None,
    ) -> list:
        """Build the time-dependent Hamiltonian."""
        pulse_func = get_pulse(pulse_name)
        detuning_func = get_detuning(detuning_name)
        phase_func = get_phase(phase_name)
        
        pulse_kwargs = pulse_kwargs or {}
        detuning_kwargs = detuning_kwargs or {}
        phase_kwargs = phase_kwargs or {}
        
        # Coefficient functions for QuTiP
        def omega_coeff(t, args):
            return pulse_func(
                t,
                args['t_center'],
                args['omega'],
                args['sweep_time'],
                **pulse_kwargs
            )
        
        def delta_coeff(t, args):
            return detuning_func(
                t= t,
                t_center= args['t_center'],
                freq_span= args['freq_span'],
                freq_span_center= args['freq_span_center'],
                sweep_time= args['sweep_time'],
                **detuning_kwargs
            )
        def phase_coeff(t, args):
            return phase_func(
                t,
                args['t_center'],
                args['phase'],
                args['sweep_time'],
                **phase_kwargs
            )

        def omega_x_coeff(t, args):
            omega = omega_coeff(t, args)
            phase = phase_coeff(t, args)
            return omega * np.cos(phase)
        
        def omega_y_coeff(t, args):
            omega = omega_coeff(t,args)
            phase = phase_coeff(t, args)
            return omega * np.sin(phase)
        
        # Hamiltonian: H = (Ω/2)σx + Δσz
        H_x = 0.5 * sigmax()
        H_y = 0.5 * sigmay()
        H_delta = sigmaz()
        
        return [[H_x, omega_x_coeff], [H_y, omega_y_coeff], [H_delta, delta_coeff]]
    
    def _compute_bloch_coords(
        self,
        c0: np.ndarray,
        c1: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute Bloch sphere coordinates from state amplitudes."""
        # Convert to numpy arrays if needed
        c0 = np.array([complex(c) for c in c0])
        c1 = np.array([complex(c) for c in c1])
        
        # Bloch sphere angles
        theta = 2 * np.arctan2(np.abs(c1), np.abs(c0))
        phi = np.pi + np.angle(c1) - np.angle(c0)
        
        # Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return {'x': x, 'y': y, 'z': z}
    
    def run(
        self,
        pulse: str = "constant",
        detuning: str = "linear",
        phase: str = "constant",
        initial_state: Qobj | None = None,
        pulse_kwargs: dict | None = None,
        detuning_kwargs: dict | None = None,
        phase_kwargs: dict | None = None,
    ) -> SimulationResult:
        """
        Run the composite pulse simulation.
        
        Args:
            pulse: Name of the pulse shape (default: "constant").
            detuning: Name of the detuning profile (default: "linear").
            phase: Name of the phase profile (default: "constant").
            initial_state: Initial quantum state. Defaults to |0⟩.
            pulse_kwargs: Additional kwargs for the pulse function.
            detuning_kwargs: Additional kwargs for the detuning function.
            
        Returns:
            SimulationResult containing all simulation data.
        """
        if initial_state is None:
            initial_state = self._psi0
        
        # Build Hamiltonian
        H = self._build_hamiltonian(pulse, detuning, phase, pulse_kwargs, detuning_kwargs, phase_kwargs)
        
        # Time array
        n_points = int(self.params.T / self.params.dt)
        tlist = np.linspace(0, self.params.T, n_points)
        
        # Solve Schrödinger equation
        args = self.params.to_dict()
        result = sesolve(H, initial_state, tlist, args=args)
        
        # Extract state amplitudes
        c0 = np.array([self._psi0.dag() * state for state in result.states])
        c1 = np.array([self._psi1.dag() * state for state in result.states])
        
        # Probabilities
        p0 = np.abs(c0)**2
        p1 = np.abs(c1)**2
        
        # Bloch coordinates
        bloch_coords = self._compute_bloch_coords(c0, c1)
        
        return SimulationResult(
            times=tlist,
            states=result.states,
            c0=c0,
            c1=c1,
            probabilities={'p0': p0, 'p1': p1},
            bloch_coords=bloch_coords,
            params=self.params,
            pulse_name=pulse,
            detuning_name=detuning,
            phase_name=phase,
            qutip_result=result,
        )
    
    def get_pulse_profile(self, pulse: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the pulse amplitude over time.
        
        Returns:
            Tuple of (times, amplitudes) arrays.
        """
        pulse_func = get_pulse(pulse)
        n_points = int(self.params.T / self.params.dt)
        times = np.linspace(0, self.params.T, n_points)
        
        amplitudes = np.array([
            pulse_func(t, self.params.t_center, self.params.omega, 
                      self.params.sweep_time, **kwargs)
            for t in times
        ])
        
        return times, amplitudes
    
    def get_detuning_profile(self, detuning: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the detuning over time.
        
        Returns:
            Tuple of (times, detunings) arrays.
        """
        detuning_func = get_detuning(detuning)
        n_points = int(self.params.T / self.params.dt)
        times = np.linspace(0, self.params.T, n_points)
        
        detunings = np.array([
            detuning_func(t, self.params.t_center, self.params.delta_span,
                         self.params.span_center, self.params.sweep_time, **kwargs)
            for t in times
        ])
        return times, detunings

    def get_phase_profile(self, phase: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the phase over time.
        
        Returns:
            Tuple of (times, detunings) arrays.
        """
        phase_func = get_phase(phase)
        n_points = int(self.params.T / self.params.dt)
        times = np.linspace(0, self.params.T, n_points)
        
        phases = np.array([
            phase_func(t, self.params.t_center, self.params.phase, self.params.sweep_time, **kwargs)
            for t in times
        ])
        return times, phases

    

