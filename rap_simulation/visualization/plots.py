"""
Static plotting functions for RAP simulation results.

Uses matplotlib for plotting and integrates with QuTiP's expect() function
for calculating expectation values.
"""

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from qutip import sigmax, sigmay, sigmaz, expect

if TYPE_CHECKING:
    from ..simulation import SimulationResult
    from ..analysis import AdiabaticityAnalysis


def plot_probabilities(
    result: "SimulationResult",
    time_unit: str = "ms",
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> Figure:
    """
    Plot state probabilities over time.
    
    Args:
        result: Simulation result object.
        time_unit: Time unit for x-axis ('s', 'ms', 'us').
        figsize: Figure size in inches.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    time_scale = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(time_unit, 1)
    times = result.times * time_scale
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, result.p0, label=r'$P_{|0\rangle}$', color='#2ecc71', linewidth=2)
    ax.plot(times, result.p1, label=r'$P_{|1\rangle}$', color='#e74c3c', linewidth=2)
    
    ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=11, loc='center right')
    ax.set_title(f'State Populations (Transfer: {result.final_p1:.3f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_pulse_detuning_and_phase(
    result: "SimulationResult",
    time_unit: str = "ms",
    freq_unit: str = "kHz",
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> Figure:
    """
    Plot the Rabi frequency (pulse) and detuning profiles.
    
    Args:
        result: Simulation result object.
        time_unit: Time unit for x-axis.
        freq_unit: Frequency unit for y-axis ('Hz', 'kHz', 'MHz').
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    from ..pulses import get_pulse
    from ..detuning import get_detuning
    from ..phase import get_phase
    
    time_scale = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(time_unit, 1)
    freq_scale = {'Hz': 1, 'kHz': 1e-3, 'MHz': 1e-6}.get(freq_unit, 1)
    
    times = result.times * time_scale
    params = result.params
    dic_params = {"sweep_time": params.sweep_time, 
                  "t_center" : params.t_center, 
                  "omega" : params.omega, 
                  "theta" : params.theta,
                  "omega_eff" : params.omega_eff,
                  } 
    
    # Get pulse and detuning functions
    pulse_func = get_pulse(result.pulse_name)
    detuning_func = get_detuning(result.detuning_name)
    phase_func = get_phase(result.phase_name)
    
    # Calculate profiles
    omega = np.array([
        pulse_func(t, params.t_center, params.omega, params.sweep_time)
        for t in result.times
    ]) / (2 * np.pi) * freq_scale
    
    delta = np.array([
        detuning_func(t, params.t_center, params.freq_span, params.freq_span_center, params.sweep_time)
        for t in result.times
    ]) / (2 * np.pi) * freq_scale

    phi = np.array([
        phase_func(t, dic_params)
        for t in result.times
    ])

    
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, omega, label=r'$\Omega(t)$ (Rabi freq.)', color='#3498db', linewidth=2)
    ax.plot(times, delta, label=r'$\Delta(t)$ (Detuning)', color='#9b59b6', linewidth=2)
    ax.plot(times, phi, label=r'$\Phi(t)$ (Phase)', color="#e97400", linewidth=2)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
    ax.set_ylabel(f'Frequency ({freq_unit})', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title(f'Pulse: {result.pulse_name}, Detuning: {result.detuning_name}', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_adiabaticity(
    analysis: "AdiabaticityAnalysis",
    time_unit: str = "ms",
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> Figure:
    """
    Plot the adiabaticity criterion over time.
    
    Args:
        analysis: AdiabaticityAnalysis object from adiabaticity_criterion().
        time_unit: Time unit for x-axis.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    time_scale = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(time_unit, 1)
    times = analysis.times * time_scale
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, np.abs(analysis.criterion), color='#e67e22', linewidth=2)
    ax.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Adiabatic threshold')
    
    ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
    ax.set_ylabel(r'$|d\theta/dt| / \Omega_{eff}$', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    
    status = "ADIABATIC" if analysis.is_adiabatic else "NON-ADIABATIC"
    ax.set_title(f'Adiabaticity Criterion ({status})', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_expectation_values(
    result: "SimulationResult",
    operators: list | None = None,
    time_unit: str = "ms",
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    Plot expectation values of spin operators over time.
    
    Uses QuTiP's expect() function to compute expectation values.
    
    Args:
        result: Simulation result object.
        operators: List of QuTiP operators. Defaults to [σx, σy, σz].
        time_unit: Time unit for x-axis.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    time_scale = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(time_unit, 1)
    times = result.times * time_scale
    
    if operators is None:
        operators = [sigmax(), sigmay(), sigmaz()]
        labels = [r'$\langle\sigma_x\rangle$', r'$\langle\sigma_y\rangle$', r'$\langle\sigma_z\rangle$']
        colors = ['#e74c3c', '#2ecc71', '#3498db']
    else:
        labels = [f'Op {i}' for i in range(len(operators))]
        colors = plt.cm.Set1(np.linspace(0, 1, len(operators)))
    
    # Use QuTiP's expect function
    expectations = expect(operators, result.states)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for exp, label, color in zip(expectations, labels, colors):
        ax.plot(times, exp, label=label, linewidth=2, color=color)
    
    ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
    ax.set_ylabel('Expectation value', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='center right')
    ax.set_title('Spin Expectation Values', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_combined(
    result: "SimulationResult",
    analysis: "AdiabaticityAnalysis | None" = None,
    time_unit: str = "ms",
    figsize: tuple[float, float] = (12, 10),
    show: bool = True,
) -> Figure:
    """
    Create a combined plot with probabilities, pulse/detuning, and optionally adiabaticity.
    
    Args:
        result: Simulation result object.
        analysis: Optional AdiabaticityAnalysis object.
        time_unit: Time unit for x-axis.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    from ..pulses import get_pulse
    from ..detuning import get_detuning
    
    time_scale = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(time_unit, 1)
    times = result.times * time_scale
    params = result.params
    
    n_rows = 3 if analysis is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    
    # Panel 1: Probabilities
    ax1 = axes[0]
    ax1.plot(times, result.p0, label=r'$P_{|0\rangle}$', color='#2ecc71', linewidth=2)
    ax1.plot(times, result.p1, label=r'$P_{|1\rangle}$', color='#e74c3c', linewidth=2)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(fontsize=10, loc='center right')
    ax1.set_title(f'Rapid Adiabatic Passage (Transfer: {result.final_p1:.3f})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Pulse and Detuning
    ax2 = axes[1]
    pulse_func = get_pulse(result.pulse_name)
    detuning_func = get_detuning(result.detuning_name)
    
    omega = np.array([
        pulse_func(t, params.t_center, params.omega, params.sweep_time)
        for t in result.times
    ]) / (2 * np.pi * 1e3)  # Convert to kHz
    
    delta = np.array([
        detuning_func(t, params.t_center, params.freq_span, params.freq_span_center, params.sweep_time)
        for t in result.times
    ]) / (2 * np.pi * 1e3)  # Convert to kHz
    
    ax2.plot(times, omega, label=r'$\Omega(t)$', color='#3498db', linewidth=2)
    ax2.plot(times, delta, label=r'$\Delta(t)$', color='#9b59b6', linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Frequency (kHz)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Adiabaticity (if provided)
    if analysis is not None:
        ax3 = axes[2]
        ax3.semilogy(times, np.abs(analysis.criterion), color='#e67e22', linewidth=2)
        ax3.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax3.set_ylabel('Adiabaticity', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel(f'Time ({time_unit})', fontsize=11)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_sweep_results(
    parameter_values: np.ndarray,
    p0_values: np.ndarray,
    p1_values: np.ndarray,
    parameter_name: str = "Parameter",
    parameter_unit: str = "",
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> Figure:
    """
    Plot results from a parameter sweep.
    
    Args:
        parameter_values: Array of parameter values.
        p0_values: Final P0 for each parameter value.
        p1_values: Final P1 for each parameter value.
        parameter_name: Name of the swept parameter.
        parameter_unit: Unit of the parameter.
        figsize: Figure size.
        show: Whether to call plt.show().
        
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(parameter_values, p0_values, 'o-', label=r'$P_{|0\rangle}$', 
            color='#2ecc71', markersize=4, linewidth=1.5)
    ax.plot(parameter_values, p1_values, 'o-', label=r'$P_{|1\rangle}$',
            color='#e74c3c', markersize=4, linewidth=1.5)
    
    # Mark optimal point
    optimal_idx = np.argmax(p1_values)
    optimal_param = parameter_values[optimal_idx]
    optimal_p1 = p1_values[optimal_idx]
    
    ax.axvline(optimal_param, color='gray', linestyle='--', alpha=0.5)
    ax.plot(optimal_param, optimal_p1, 'k*', markersize=15, 
            label=f'Optimal: {optimal_param:.3g}')
    
    xlabel = f'{parameter_name} ({parameter_unit})' if parameter_unit else parameter_name
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Final Probability', fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=11)
    ax.set_title(f'{parameter_name} Sweep', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig

