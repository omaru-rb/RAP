"""
Parameter sweep utilities for rapid adiabatic passage simulations.

This module provides functions to scan simulation parameters and find
optimal conditions for population transfer.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from .atoms.base import Atom
    from .simulation import SimulationParams


@dataclass
class SweepResult:
    """
    Results from a parameter sweep.
    
    Attributes:
        parameter_name: Name of the swept parameter.
        parameter_values: Array of parameter values.
        p0_final: Final P0 probability for each parameter value.
        p1_final: Final P1 probability for each parameter value.
        optimal_value: Parameter value giving maximum P1.
        optimal_p1: Maximum P1 achieved.
    """
    parameter_name: str
    parameter_values: np.ndarray
    p0_final: np.ndarray
    p1_final: np.ndarray
    
    @property
    def optimal_index(self) -> int:
        """Index of optimal parameter value."""
        return int(np.argmax(self.p1_final))
    
    @property
    def optimal_value(self) -> float:
        """Optimal parameter value."""
        return float(self.parameter_values[self.optimal_index])
    
    @property
    def optimal_p1(self) -> float:
        """Maximum P1 achieved."""
        return float(self.p1_final[self.optimal_index])


def sweep_time_scan(
    atom: "Atom",
    base_params: "SimulationParams",
    t_start: float,
    t_end: float,
    n_points: int,
    pulse: str = "hyper_secant",
    detuning: str = "linear",
    constant_freq_span: float | None = None,
    show_progress: bool = True,
) -> SweepResult:
    """
    Scan the sweep time to find optimal adiabatic passage duration.
    
    Args:
        atom: Atom object.
        base_params: Base simulation parameters.
        t_start: Starting sweep time (s).
        t_end: Ending sweep time (s).
        n_points: Number of points in the scan.
        pulse: Pulse shape name.
        detuning: Detuning profile name.
        constant_freq_span: If provided, keep total frequency span constant (Hz).
                           Otherwise, delta_span is recalculated to maintain the same rate.
        show_progress: Whether to show a progress bar.
        
    Returns:
        SweepResult with scan data.
    """
    from .simulation import RapidAdiabaticPassage, SimulationParams
    
    sweep_times = np.linspace(t_start, t_end, n_points)
    p0_final = np.zeros(n_points)
    p1_final = np.zeros(n_points)
    
    iterator = tqdm(enumerate(sweep_times), total=n_points, desc="Sweep time scan") if show_progress else enumerate(sweep_times)
    
    for i, sweep_time in iterator:
        # Create modified parameters
        if constant_freq_span is not None:
            # Keep total frequency span constant
            delta_span = 2 * np.pi * constant_freq_span / (2 * sweep_time)
        else:
            # Scale delta_span with sweep_time to maintain the same rate
            delta_span = base_params.delta_span * (base_params.sweep_time / sweep_time)
        
        # Ensure T is long enough for the sweep
        T = max(base_params.T, 4 * sweep_time)
        
        params = SimulationParams(
            T=T,
            dt=base_params.dt,
            omega=base_params.omega,
            sweep_time=sweep_time,
            delta_span=delta_span,
            span_center=base_params.span_center,
        )
        
        rap = RapidAdiabaticPassage(atom, params)
        result = rap.run(pulse=pulse, detuning=detuning)
        
        p0_final[i] = result.probabilities['p0'][-1]
        p1_final[i] = result.probabilities['p1'][-1]
    
    return SweepResult(
        parameter_name="sweep_time",
        parameter_values=sweep_times,
        p0_final=p0_final,
        p1_final=p1_final,
    )


def frequency_span_scan(
    atom: "Atom",
    base_params: "SimulationParams",
    f_start: float,
    f_end: float,
    n_points: int,
    pulse: str = "hyper_secant",
    detuning: str = "linear",
    show_progress: bool = True,
) -> SweepResult:
    """
    Scan the frequency span to find optimal sweep range.
    
    Args:
        atom: Atom object.
        base_params: Base simulation parameters.
        f_start: Starting frequency span (Hz).
        f_end: Ending frequency span (Hz).
        n_points: Number of points in the scan.
        pulse: Pulse shape name.
        detuning: Detuning profile name.
        show_progress: Whether to show a progress bar.
        
    Returns:
        SweepResult with scan data.
    """
    from .simulation import RapidAdiabaticPassage, SimulationParams
    
    freq_spans = np.linspace(f_start, f_end, n_points)
    p0_final = np.zeros(n_points)
    p1_final = np.zeros(n_points)
    
    iterator = tqdm(enumerate(freq_spans), total=n_points, desc="Frequency span scan") if show_progress else enumerate(freq_spans)
    
    for i, freq_span in iterator:
        delta_span = 2 * np.pi * freq_span / (2 * base_params.sweep_time)
        
        params = SimulationParams(
            T=base_params.T,
            dt=base_params.dt,
            omega=base_params.omega,
            sweep_time=base_params.sweep_time,
            delta_span=delta_span,
            span_center=base_params.span_center,
        )
        
        rap = RapidAdiabaticPassage(atom, params)
        result = rap.run(pulse=pulse, detuning=detuning)
        
        p0_final[i] = result.probabilities['p0'][-1]
        p1_final[i] = result.probabilities['p1'][-1]
    
    return SweepResult(
        parameter_name="frequency_span",
        parameter_values=freq_spans,
        p0_final=p0_final,
        p1_final=p1_final,
    )


def spectroscopy_scan(
    atom: "Atom",
    base_params: "SimulationParams",
    center_start: float,
    center_end: float,
    n_points: int,
    pulse: str = "hyper_secant",
    detuning: str = "linear",
    show_progress: bool = True,
) -> SweepResult:
    """
    Perform spectroscopy by scanning the center frequency.
    
    This simulates scanning the microwave/RF frequency across the atomic
    resonance, as would be done in a clock or magnetometer experiment.
    
    Args:
        atom: Atom object.
        base_params: Base simulation parameters.
        center_start: Starting center frequency offset (Hz).
        center_end: Ending center frequency offset (Hz).
        n_points: Number of points in the scan.
        pulse: Pulse shape name.
        detuning: Detuning profile name.
        show_progress: Whether to show a progress bar.
        
    Returns:
        SweepResult with scan data.
    """
    from .simulation import RapidAdiabaticPassage, SimulationParams
    
    center_freqs = np.linspace(center_start, center_end, n_points)
    p0_final = np.zeros(n_points)
    p1_final = np.zeros(n_points)
    
    iterator = tqdm(enumerate(center_freqs), total=n_points, desc="Spectroscopy scan") if show_progress else enumerate(center_freqs)
    
    for i, center_freq in iterator:
        span_center = 2 * np.pi * center_freq
        
        params = SimulationParams(
            T=base_params.T,
            dt=base_params.dt,
            omega=base_params.omega,
            sweep_time=base_params.sweep_time,
            delta_span=base_params.delta_span,
            span_center=span_center,
        )
        
        rap = RapidAdiabaticPassage(atom, params)
        result = rap.run(pulse=pulse, detuning=detuning)
        
        p0_final[i] = result.probabilities['p0'][-1]
        p1_final[i] = result.probabilities['p1'][-1]
    
    return SweepResult(
        parameter_name="center_frequency",
        parameter_values=center_freqs,
        p0_final=p0_final,
        p1_final=p1_final,
    )


def rabi_frequency_scan(
    atom: "Atom",
    base_params: "SimulationParams",
    omega_start: float,
    omega_end: float,
    n_points: int,
    pulse: str = "hyper_secant",
    detuning: str = "linear",
    show_progress: bool = True,
) -> SweepResult:
    """
    Scan the Rabi frequency to study power dependence.
    
    Args:
        atom: Atom object.
        base_params: Base simulation parameters.
        omega_start: Starting Rabi frequency in Hz.
        omega_end: Ending Rabi frequency in Hz.
        n_points: Number of points in the scan.
        pulse: Pulse shape name.
        detuning: Detuning profile name.
        show_progress: Whether to show a progress bar.
        
    Returns:
        SweepResult with scan data.
    """
    from .simulation import RapidAdiabaticPassage, SimulationParams
    
    rabi_freqs = np.linspace(omega_start, omega_end, n_points)
    p0_final = np.zeros(n_points)
    p1_final = np.zeros(n_points)
    
    iterator = tqdm(enumerate(rabi_freqs), total=n_points, desc="Rabi frequency scan") if show_progress else enumerate(rabi_freqs)
    
    for i, rabi_freq in iterator:
        omega = 2 * np.pi * rabi_freq
        
        params = SimulationParams(
            T=base_params.T,
            dt=base_params.dt,
            omega=omega,
            sweep_time=base_params.sweep_time,
            delta_span=base_params.delta_span,
            span_center=base_params.span_center,
        )
        
        rap = RapidAdiabaticPassage(atom, params)
        result = rap.run(pulse=pulse, detuning=detuning)
        
        p0_final[i] = result.probabilities['p0'][-1]
        p1_final[i] = result.probabilities['p1'][-1]
    
    return SweepResult(
        parameter_name="rabi_frequency",
        parameter_values=rabi_freqs,
        p0_final=p0_final,
        p1_final=p1_final,
    )


def generic_parameter_scan(
    atom: "Atom",
    base_params: "SimulationParams",
    parameter_name: str,
    values: np.ndarray,
    pulse: str = "hyper_secant",
    detuning: str = "linear",
    param_modifier: Callable[["SimulationParams", float], "SimulationParams"] | None = None,
    show_progress: bool = True,
) -> SweepResult:
    """
    Generic parameter scan for custom parameters.
    
    Args:
        atom: Atom object.
        base_params: Base simulation parameters.
        parameter_name: Name of parameter being scanned (for labeling).
        values: Array of parameter values to scan.
        pulse: Pulse shape name.
        detuning: Detuning profile name.
        param_modifier: Function that takes (params, value) and returns modified params.
                       If None, will try to set attribute directly on params.
        show_progress: Whether to show a progress bar.
        
    Returns:
        SweepResult with scan data.
        
    Example:
        >>> def modify_params(params, value):
        ...     return SimulationParams(
        ...         T=params.T,
        ...         dt=value,  # Scanning dt
        ...         omega=params.omega,
        ...         ...
        ...     )
        >>> result = generic_parameter_scan(atom, params, "dt", values, param_modifier=modify_params)
    """
    from .simulation import RapidAdiabaticPassage, SimulationParams
    from dataclasses import replace
    
    n_points = len(values)
    p0_final = np.zeros(n_points)
    p1_final = np.zeros(n_points)
    
    iterator = tqdm(enumerate(values), total=n_points, desc=f"Scanning {parameter_name}") if show_progress else enumerate(values)
    
    for i, value in iterator:
        if param_modifier is not None:
            params = param_modifier(base_params, value)
        else:
            # Try to replace the attribute directly
            params = replace(base_params, **{parameter_name: value})
        
        rap = RapidAdiabaticPassage(atom, params)
        result = rap.run(pulse=pulse, detuning=detuning)
        
        p0_final[i] = result.probabilities['p0'][-1]
        p1_final[i] = result.probabilities['p1'][-1]
    
    return SweepResult(
        parameter_name=parameter_name,
        parameter_values=values,
        p0_final=p0_final,
        p1_final=p1_final,
    )


def compare_pulse_shapes(
    atom: "Atom",
    base_params: "SimulationParams",
    pulse_names: list[str] | None = None,
    detuning: str = "linear",
) -> dict[str, float]:
    """
    Compare different pulse shapes for the same parameters.
    
    Args:
        atom: Atom object.
        base_params: Simulation parameters.
        pulse_names: List of pulse names to compare. If None, uses all registered.
        detuning: Detuning profile to use.
        
    Returns:
        Dictionary mapping pulse name to final P1.
    """
    from .simulation import RapidAdiabaticPassage
    from .pulses import list_pulses
    
    if pulse_names is None:
        pulse_names = list_pulses()
    
    results = {}
    
    for pulse_name in pulse_names:
        rap = RapidAdiabaticPassage(atom, base_params)
        result = rap.run(pulse=pulse_name, detuning=detuning)
        results[pulse_name] = result.final_p1
    
    return results


def compare_detuning_profiles(
    atom: "Atom",
    base_params: "SimulationParams",
    pulse: str = "hyper_secant",
    detuning_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compare different detuning profiles for the same parameters.
    
    Args:
        atom: Atom object.
        base_params: Simulation parameters.
        pulse: Pulse shape to use.
        detuning_names: List of detuning names. If None, uses all registered.
        
    Returns:
        Dictionary mapping detuning name to final P1.
    """
    from .simulation import RapidAdiabaticPassage
    from .detuning import list_detunings
    
    if detuning_names is None:
        detuning_names = list_detunings()
    
    results = {}
    
    for detuning_name in detuning_names:
        rap = RapidAdiabaticPassage(atom, base_params)
        result = rap.run(pulse=pulse, detuning=detuning_name)
        results[detuning_name] = result.final_p1
    
    return results

