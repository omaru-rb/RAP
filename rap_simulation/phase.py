"""
Phase shape functions for Rabi frequency modulation.

This module provides various pulse shapes used to control the Rabi frequency
during rapid adiabatic passage experiments. Pulse shapes are registered
using a decorator pattern for easy extensibility.
"""

from typing import Callable, Protocol
import numpy as np


class PhaseFunc(Protocol):
    """Protocol for pulse shape functions."""
    
    def __call__(
        self,
        t: float,
        args: dict,
        **kwargs
    ) -> float:
        """
        Calculate pulse amplitude at time t.
        
        Args:
            t: Current time (s).
            t_center: Center of the pulse (s).
            phase: Peak Rabi frequency (rad/s).
            sweep_time: Half-duration of the sweep (s).
            **kwargs: Additional pulse-specific parameters.
            
        Returns:
            Rabi frequency at time t (rad/s).
        """
        ...


# Registry for pulse shapes
_PHASE_REGISTRY: dict[str, PhaseFunc] = {}


def register_phase(name: str) -> Callable[[PhaseFunc], PhaseFunc]:
    """
    Decorator to register a pulse shape function.
    
    Args:
        name: Name to register the pulse under.
        
    Example:
        @register_pulse("my_pulse")
        def my_pulse_shape(t, t_center, omega, sweep_time):
            ...
    """
    def decorator(func: PhaseFunc) -> PhaseFunc:
        _PHASE_REGISTRY[name] = func
        return func
    return decorator


def get_phase(name: str) -> PhaseFunc:
    """
    Get a registered pulse shape function by name.
    
    Args:
        name: Name of the registered pulse.
        
    Returns:
        The pulse shape function.
        
    Raises:
        KeyError: If pulse name is not registered.
    """
    if name not in _PHASE_REGISTRY:
        available = ", ".join(_PHASE_REGISTRY.keys())
        raise KeyError(f"Unknown phase '{name}'. Available: {available}")
    return _PHASE_REGISTRY[name]


def list_phase() -> list[str]:
    """Return list of registered phase names."""
    return list(_PHASE_REGISTRY.keys())


# ============================================================================
# Built-in phase profiles
# ============================================================================

@register_phase("constant")
def phase_constant(
    t: float,
    args: dict,
    **kwargs
) -> float:
    """
    Constant (rectangular) phase profile.
    
    """
    return np.pi/2

@register_phase("BB1")
def phase_BB1(
    t: float,
    args: dict,
    **kwargs
) -> float:
    """
    BB1 phase profile. Consists of 4 jumps in 
    
    """
    phi=np.arccos(-args['theta']/(4*np.pi))
    omega=args['omega']
    Ttheta=args['theta']/omega + args['overshoot']/5
    t_center=args['t_center']
    sweep_time=args['sweep_time']
    Tpi=np.pi/omega + args['overshoot']/5
    t0 = t_center - (sweep_time)/2
    t1 = t0 + Ttheta
    t2 = t1 + Tpi
    t3 = t2 + 2*Tpi
    t4 = t3 + Tpi

    if t < t1:
        return 0
    elif t < t2:
        return phi
    elif t < t3:
        return 3*phi
    elif t < t4:
        return phi
    else:
        return 0
    
    



# ============================================================================
# QuTiP-compatible coefficient functions
# ============================================================================

def make_phase_coefficients(
    phase_name: str,
    t_center: float,
    phase: float,
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
    phase_func = get_phase(phase_name)
    
    def coefficient(t: float, args: dict) -> float:
        return phase_func(
            t,
            args,
            **kwargs
        )
    
    return coefficient

