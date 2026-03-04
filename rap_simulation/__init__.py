"""
RAP Simulation - A package for simulating Rapid Adiabatic Passage in atomic systems.

This package provides tools for simulating and visualizing rapid adiabatic passage
experiments using QuTiP for quantum dynamics calculations.
"""

from .atoms import Atom, Rubidium87, HyperfineState, Transition
from .simulation import CompositePulse, SimulationParams, SimulationResult
from .pulses import get_pulse, list_pulses
from .detuning import get_detuning, list_detunings
from .phase import get_phase, list_phase
from .analysis import adiabaticity_criterion, compute_bloch_trajectory
from .scans import sweep_time_scan, frequency_span_scan, spectroscopy_scan

__version__ = "0.1.0"

__all__ = [
    # Atoms
    "Atom",
    "Rubidium87",
    "HyperfineState",
    "Transition",
    # Simulation
    "CompositePulse",
    "SimulationParams",
    "SimulationResult",
    # Pulses & Detuning
    "get_pulse",
    "list_pulses",
    "get_detuning",
    "list_detunings",
    "get_phase",
    "list_phase",
    # Analysis
    "adiabaticity_criterion",
    "compute_bloch_trajectory",
    # Scans
    "sweep_time_scan",
    "frequency_span_scan",
    "spectroscopy_scan",
]

