"""
Pulse shape functions for Rabi frequency modulation.

This module provides various pulse shapes used to control the Rabi frequency
during rapid adiabatic passage experiments. Pulse shapes are registered
using a decorator pattern for easy extensibility.
"""

from typing import Callable, Protocol
import numpy as np


class PulseFunc(Protocol):
    """Protocol for pulse shape functions."""
    
    def __call__(
        self,
        t: float,
        t_center: float,
        omega: float,
        sweep_time: float,
        **kwargs
    ) -> float:
        """
        Calculate pulse amplitude at time t.
        
        Args:
            t: Current time (s).
            t_center: Center of the pulse (s).
            omega: Peak Rabi frequency (rad/s).
            sweep_time: Half-duration of the sweep (s).
            **kwargs: Additional pulse-specific parameters.
            
        Returns:
            Rabi frequency at time t (rad/s).
        """
        ...


# Registry for pulse shapes
_PULSE_REGISTRY: dict[str, PulseFunc] = {}


def register_pulse(name: str) -> Callable[[PulseFunc], PulseFunc]:
    """
    Decorator to register a pulse shape function.
    
    Args:
        name: Name to register the pulse under.
        
    Example:
        @register_pulse("my_pulse")
        def my_pulse_shape(t, t_center, omega, sweep_time):
            ...
    """
    def decorator(func: PulseFunc) -> PulseFunc:
        _PULSE_REGISTRY[name] = func
        return func
    return decorator


def get_pulse(name: str) -> PulseFunc:
    """
    Get a registered pulse shape function by name.
    
    Args:
        name: Name of the registered pulse.
        
    Returns:
        The pulse shape function.
        
    Raises:
        KeyError: If pulse name is not registered.
    """
    if name not in _PULSE_REGISTRY:
        available = ", ".join(_PULSE_REGISTRY.keys())
        raise KeyError(f"Unknown pulse '{name}'. Available: {available}")
    return _PULSE_REGISTRY[name]


def list_pulses() -> list[str]:
    """Return list of registered pulse names."""
    return list(_PULSE_REGISTRY.keys())


# ============================================================================
# Built-in pulse shapes
# ============================================================================

@register_pulse("constant")
def pulse_constant(
    t: float,
    t_center: float,
    omega: float,
    sweep_time: float,
    **kwargs
) -> float:
    """
    Constant (rectangular) pulse shape.
    
    Returns omega during the sweep window, and a small baseline value outside.
    """
    if abs(t - t_center) > sweep_time:
        return 1.0  # Small baseline to avoid division issues
    return omega


@register_pulse("gaussian")
def pulse_gaussian(
    t: float,
    t_center: float,
    omega: float,
    sweep_time: float,
    sigma: float | None = None,
    **kwargs
) -> float:
    """
    Gaussian pulse shape.
    
    Args:
        sigma: Width parameter. Defaults to sweep_time/3 for ~99% within window.
    """
    if sigma is None:
        sigma = sweep_time / 3
        
    if abs(t - t_center) > sweep_time:
        return 1.0
    return omega * np.exp(-(t - t_center)**2 / (2 * sigma**2))


@register_pulse("lorentzian")
def pulse_lorentzian(
    t: float,
    t_center: float,
    omega: float,
    sweep_time: float,
    gamma: float | None = None,
    **kwargs
) -> float:
    """
    Lorentzian pulse shape.
    
    Args:
        gamma: Half-width at half-maximum. Defaults to 0.1 * sweep_time.
    """
    if gamma is None:
        gamma = 0.1 * sweep_time
        
    if abs(t - t_center) > sweep_time:
        return 1.0
    return omega * gamma**2 / ((t - t_center)**2 + gamma**2)


@register_pulse("hyper_secant")
def pulse_hyper_secant(
    t: float,
    t_center: float,
    omega: float,
    sweep_time: float,
    beta: float = 5.3,
    **kwargs
) -> float:
    """
    Hyperbolic secant pulse shape.
    
    This is the optimal pulse shape for adiabatic passage as it provides
    smooth turn-on/off with minimal non-adiabatic transitions.
    
    Args:
        beta: Shape parameter controlling the steepness (default: 5.3).
    """
    if abs(t - t_center) > sweep_time:
        return 0.1  # Small but non-zero baseline
    return omega / np.cosh(beta * (t - t_center) / sweep_time)


# ============================================================================
# QuTiP-compatible coefficient functions
# ============================================================================

def make_pulse_coefficients(
    pulse_name: str,
    t_center: float,
    omega: float,
    sweep_time: float,
    **kwargs
) -> Callable[[float, dict], float]:
    """
    Create a QuTiP-compatible coefficient function for the given pulse.
    
    Args:
        pulse_name: Name of the registered pulse.
        t_center: Center of the pulse (s).
        omega: Peak Rabi frequency (rad/s).
        sweep_time: Half-duration of the sweep (s).
        **kwargs: Additional pulse parameters.
        
    Returns:
        A function f(t, args) suitable for QuTiP's time-dependent Hamiltonians.
    """
    pulse_func = get_pulse(pulse_name)
    
    def coefficient(t: float, args: dict) -> float:
        return pulse_func(
            t,
            args.get('t_center', t_center),
            args.get('omega', omega),
            args.get('sweep_time', sweep_time),
            **kwargs
        )
    
    return coefficient

