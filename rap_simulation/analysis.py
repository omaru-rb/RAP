"""
Analysis functions for rapid adiabatic passage simulations.

This module provides tools for analyzing simulation results, including
adiabaticity criteria calculations and Bloch sphere trajectory analysis.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from .pulses import get_pulse
from .detuning import get_detuning
from.phase import get_phase

if TYPE_CHECKING:
    from .simulation import SimulationParams, SimulationResult


@dataclass
class AdiabaticityAnalysis:
    """
    Results from adiabaticity analysis.
    
    Attributes:
        times: Time array (s).
        criterion: Adiabaticity parameter at each time (should be << 1).
        omega: Rabi frequency profile (rad/s).
        delta: Detuning profile (rad/s).
        driving_angle: Angle of effective field in x-z plane (rad).
        driving_angle_rate: Rate of change of driving angle (rad/s).
        effective_frequency: Effective Rabi frequency sqrt(Ω² + Δ²) (rad/s).
        is_adiabatic: Boolean indicating if criterion < threshold throughout.
    """
    times: np.ndarray
    criterion: np.ndarray
    omega: np.ndarray
    delta: np.ndarray
    driving_angle: np.ndarray
    driving_angle_rate: np.ndarray
    effective_frequency: np.ndarray
    is_adiabatic: bool
    
    @property
    def max_criterion(self) -> float:
        """Maximum value of adiabaticity criterion."""
        return float(np.max(np.abs(self.criterion)))
    
    @property
    def mean_criterion(self) -> float:
        """Mean value of adiabaticity criterion."""
        return float(np.mean(np.abs(self.criterion)))


def adiabaticity_criterion(
    params: "SimulationParams",
    pulse_name: str,
    detuning_name: str,
    threshold: float = 0.1,
    pulse_kwargs: dict | None = None,
    detuning_kwargs: dict | None = None,
) -> AdiabaticityAnalysis:
    """
    Calculate the adiabaticity criterion for a given pulse/detuning combination.
    
    The adiabatic condition requires:
        |dθ/dt| << Ω_eff
    
    where θ is the angle of the effective field and Ω_eff = sqrt(Ω² + Δ²).
    
    The criterion is:
        C = |Ω̇Δ - ΔΩ̇| / (Ω² + Δ²)^(3/2)
    
    For adiabatic passage, C << 1 throughout the sweep.
    
    Args:
        params: Simulation parameters.
        pulse_name: Name of pulse shape.
        detuning_name: Name of detuning profile.
        threshold: Threshold for adiabaticity (default: 0.1).
        pulse_kwargs: Additional pulse parameters.
        detuning_kwargs: Additional detuning parameters.
        
    Returns:
        AdiabaticityAnalysis with criterion values and related quantities.
    """
    pulse_func = get_pulse(pulse_name)
    detuning_func = get_detuning(detuning_name)
    
    pulse_kwargs = pulse_kwargs or {}
    detuning_kwargs = detuning_kwargs or {}
    
    # Time array
    n_points = int(params.T / params.dt)
    times = np.linspace(0, params.T, n_points)
    dt = times[1] - times[0]
    
    # Calculate profiles
    omega = np.array([
        pulse_func(t, params.t_center, params.omega, params.sweep_time, **pulse_kwargs)
        for t in times
    ])
    
    delta = np.array([
        detuning_func(t, params.t_center, params.delta_span, params.span_center, 
                     params.sweep_time, **detuning_kwargs)
        for t in times
    ])
    
    # Derivatives
    omega_dot = np.gradient(omega, dt)
    delta_dot = np.gradient(delta, dt)
    
    # Adiabaticity criterion
    # C = |Ω̇Δ - ΔΩ̇| / (Ω² + Δ²)^(3/2)
    numerator = omega_dot * delta - delta_dot * omega
    denominator = (omega**2 + delta**2)**(3/2)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        criterion = np.where(denominator > 1e-10, numerator / denominator, 0)
    
    # Driving angle and its rate
    driving_angle = np.arctan2(omega, delta)
    driving_angle_rate = np.gradient(driving_angle, dt)
    
    # Effective frequency
    effective_frequency = np.sqrt(omega**2 + delta**2)
    
    # Check adiabaticity
    is_adiabatic = np.all(np.abs(criterion) < threshold)
    
    return AdiabaticityAnalysis(
        times=times,
        criterion=criterion,
        omega=omega,
        delta=delta,
        driving_angle=driving_angle,
        driving_angle_rate=driving_angle_rate,
        effective_frequency=effective_frequency,
        is_adiabatic=is_adiabatic,
    )


def compute_bloch_trajectory(
    result: "SimulationResult",
) -> dict[str, np.ndarray]:
    """
    Extract the Bloch sphere trajectory from simulation results.
    
    Args:
        result: Simulation result object.
        
    Returns:
        Dictionary with 'x', 'y', 'z' coordinate arrays.
    """
    return result.bloch_coords.copy()


def compute_driving_field_trajectory(
    params: "SimulationParams",
    pulse_name: str,
    detuning_name: str,
    pulse_kwargs: dict | None = None,
    detuning_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute the trajectory of the effective driving field on the Bloch sphere.
    
    The effective field direction is:
        n = (Ω, 0, Δ) / ||(Ω, 0, Δ)||
    
    In an ideal adiabatic passage, the state vector follows this field.
    
    Args:
        params: Simulation parameters.
        pulse_name: Name of pulse shape.
        detuning_name: Name of detuning profile.
        pulse_kwargs: Additional pulse parameters.
        detuning_kwargs: Additional detuning parameters.
        
    Returns:
        Dictionary with 'x', 'y', 'z' coordinate arrays of normalized field.
    """
    pulse_func = get_pulse(pulse_name)
    detuning_func = get_detuning(detuning_name)
    
    pulse_kwargs = pulse_kwargs or {}
    detuning_kwargs = detuning_kwargs or {}
    
    n_points = int(params.T / params.dt)
    times = np.linspace(0, params.T, n_points)
    
    # Calculate field components
    omega = np.array([
        pulse_func(t, params.t_center, params.omega, params.sweep_time, **pulse_kwargs)
        for t in times
    ])
    
    delta = np.array([
        detuning_func(t, params.t_center, params.delta_span, params.span_center,
                     params.sweep_time, **detuning_kwargs)
        for t in times
    ])
    
    # The driving field in the rotating frame is along (Ω, 0, -Δ) in Bloch coordinates
    # (convention: z is along the detuning axis)
    x = omega / 2  # Factor of 1/2 from Hamiltonian convention
    y = np.zeros_like(times)
    z = -delta
    
    # Normalize
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag = np.where(mag > 1e-10, mag, 1)  # Avoid division by zero
    
    return {
        'x': x / mag,
        'y': y / mag,
        'z': z / mag,
        'times': times,
    }


def state_fidelity(result: "SimulationResult", target_state: int = 1) -> float:
    """
    Calculate the final state fidelity.
    
    Args:
        result: Simulation result.
        target_state: Target state (0 or 1). Default: 1.
        
    Returns:
        Fidelity (probability of being in target state).
    """
    if target_state == 0:
        return float(result.probabilities['p0'][-1])
    elif target_state == 1:
        return float(result.probabilities['p1'][-1])
    else:
        raise ValueError(f"target_state must be 0 or 1, got {target_state}")


def transfer_efficiency(result: "SimulationResult") -> float:
    """
    Calculate the population transfer efficiency.
    
    This is simply the final probability in |1⟩ state, assuming
    we started in |0⟩.
    
    Args:
        result: Simulation result.
        
    Returns:
        Transfer efficiency (0 to 1).
    """
    return result.final_p1


def analyze_oscillations(
    result: "SimulationResult",
    state: int = 1,
) -> dict:
    """
    Analyze oscillations in the state probability.
    
    Useful for understanding the dynamics during the passage,
    especially for comparing adiabatic vs non-adiabatic regimes.
    
    Args:
        result: Simulation result.
        state: Which state probability to analyze (0 or 1).
        
    Returns:
        Dictionary with oscillation statistics.
    """
    prob_key = 'p0' if state == 0 else 'p1'
    prob = result.probabilities[prob_key]
    
    # Find local maxima and minima
    from scipy.signal import find_peaks
    
    maxima_idx, _ = find_peaks(prob)
    minima_idx, _ = find_peaks(-prob)
    
    return {
        'n_maxima': len(maxima_idx),
        'n_minima': len(minima_idx),
        'n_oscillations': min(len(maxima_idx), len(minima_idx)),
        'max_probability': float(np.max(prob)),
        'min_probability': float(np.min(prob)),
        'amplitude_range': float(np.max(prob) - np.min(prob)),
        'final_value': float(prob[-1]),
    }

