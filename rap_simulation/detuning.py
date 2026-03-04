"""
Detuning sweep functions for rapid adiabatic passage.

This module provides various detuning profiles used during RAP experiments.
The detuning represents the frequency difference between the driving field
and the atomic transition.
"""

from typing import Callable, Protocol
import numpy as np


class DetuningFunc(Protocol):
    """Protocol for detuning sweep functions."""
    
    def __call__(
        self,
        t: float,
        t_center: float,
        delta_span: float,
        span_center: float,
        sweep_time: float,
        **kwargs
    ) -> float:
        """
        Calculate detuning at time t.
        
        Args:
            t: Current time (s).
            t_center: Center of the sweep (s).
            delta_span: Rate of frequency change (rad/s per second).
            span_center: Center detuning value (rad/s).
            sweep_time: Half-duration of the sweep (s).
            **kwargs: Additional sweep-specific parameters.
            
        Returns:
            Detuning at time t (rad/s).
        """
        ...


# Registry for detuning functions
_DETUNING_REGISTRY: dict[str, DetuningFunc] = {}


def register_detuning(name: str) -> Callable[[DetuningFunc], DetuningFunc]:
    """
    Decorator to register a detuning sweep function.
    
    Args:
        name: Name to register the detuning under.
    """
    def decorator(func: DetuningFunc) -> DetuningFunc:
        _DETUNING_REGISTRY[name] = func
        return func
    return decorator


def get_detuning(name: str) -> DetuningFunc:
    """
    Get a registered detuning function by name.
    
    Args:
        name: Name of the registered detuning.
        
    Returns:
        The detuning function.
        
    Raises:
        KeyError: If detuning name is not registered.
    """
    if name not in _DETUNING_REGISTRY:
        available = ", ".join(_DETUNING_REGISTRY.keys())
        raise KeyError(f"Unknown detuning '{name}'. Available: {available}")
    return _DETUNING_REGISTRY[name]


def list_detunings() -> list[str]:
    """Return list of registered detuning names."""
    return list(_DETUNING_REGISTRY.keys())


# ============================================================================
# Built-in detuning sweeps
# ============================================================================

@register_detuning("constant")
def detuning_constant(
    t: float,
    t_center: float,
    freq: float,
    sweep_time: float,
    **kwargs
) -> float:
    """
    Constant (rectangular) detuning.
    
    """
    if abs(t - t_center) > sweep_time:
        return 0
    return freq

@register_detuning("linear")
def detuning_linear(
    t: float,
    t_center: float,
    freq_span: float,
    freq_span_center: float,
    sweep_time: float,
    **kwargs
) -> float:
    """
    Linear frequency sweep.
    
    The detuning changes linearly from -delta_span*sweep_time to 
    +delta_span*sweep_time over the sweep window.
    """
    dt = t - t_center
    slope = freq_span / sweep_time
    
    if dt < -sweep_time/2:
        return  freq_span_center - slope * sweep_time/2
    elif dt > sweep_time/2:
        return freq_span_center + slope * sweep_time/2
    else:
        return freq_span_center + slope * dt


@register_detuning("tanh")
def detuning_tanh(
    t: float,
    t_center: float,
    delta_span: float,
    span_center: float,
    sweep_time: float,
    steepness: float = 2.6,
    **kwargs
) -> float:
    """
    Hyperbolic tangent frequency sweep.
    
    This provides a smooth S-curve transition that pairs well with
    the hyperbolic secant pulse shape.
    
    Args:
        steepness: Controls the steepness of the transition (default: 2.6).
    """
    dt = t - t_center
    
    if dt < -sweep_time:
        return delta_span * (-sweep_time) + span_center
    elif dt > sweep_time:
        return delta_span * sweep_time + span_center
    else:
        return delta_span * sweep_time * np.tanh(steepness * dt / sweep_time) + span_center


@register_detuning("quadratic")
def detuning_quadratic(
    t: float,
    t_center: float,
    delta_span: float,
    span_center: float,
    sweep_time: float,
    **kwargs
) -> float:
    """
    Quadratic frequency sweep.
    
    Provides slower sweep rate near the center (resonance) which can
    improve adiabaticity near the avoided crossing.
    """
    dt = t - t_center
    
    if abs(dt) > sweep_time:
        sign = 1 if dt > 0 else -1
        return delta_span * sweep_time * sign + span_center
    else:
        # Quadratic: sign-preserving
        sign = 1 if dt >= 0 else -1
        return delta_span * (dt**2 / sweep_time) * sign + span_center


# ============================================================================
# QuTiP-compatible coefficient functions
# ============================================================================

def make_detuning_coefficients(
    detuning_name: str,
    t_center: float,
    delta_span: float,
    span_center: float,
    sweep_time: float,
    **kwargs
) -> Callable[[float, dict], float]:
    """
    Create a QuTiP-compatible coefficient function for the given detuning.
    
    Args:
        detuning_name: Name of the registered detuning.
        t_center: Center of the sweep (s).
        delta_span: Rate of frequency change (rad/s per second).
        span_center: Center detuning value (rad/s).
        sweep_time: Half-duration of the sweep (s).
        **kwargs: Additional detuning parameters.
        
    Returns:
        A function f(t, args) suitable for QuTiP's time-dependent Hamiltonians.
    """
    detuning_func = get_detuning(detuning_name)
    
    def coefficient(t: float, args: dict) -> float:
        return detuning_func(
            t,
            args.get('t_center', t_center),
            args.get('delta_span', delta_span),
            args.get('span_center', span_center),
            args.get('sweep_time', sweep_time),
            **kwargs
        )
    
    return coefficient

