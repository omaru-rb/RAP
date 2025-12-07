"""
RAP Simulation - A package for simulating Rapid Adiabatic Passage in atomic systems.

This package provides tools for simulating and visualizing rapid adiabatic passage
experiments using QuTiP for quantum dynamics calculations.
"""

from .atoms import Atom, Rubidium87
from .simulation import RapidAdiabaticPassage, SimulationParams, SimulationResult
from .pulses import get_pulse, list_pulses
from .detuning import get_detuning, list_detunings
from .analysis import adiabaticity_criterion, compute_bloch_trajectory
from .scans import sweep_time_scan, frequency_span_scan, spectroscopy_scan

__version__ = "0.1.0"

__all__ = [
    # Atoms
    "Atom",
    "Rubidium87",
    # Simulation
    "RapidAdiabaticPassage",
    "SimulationParams",
    "SimulationResult",
    # Pulses & Detuning
    "get_pulse",
    "list_pulses",
    "get_detuning",
    "list_detunings",
    # Analysis
    "adiabaticity_criterion",
    "compute_bloch_trajectory",
    # Scans
    "sweep_time_scan",
    "frequency_span_scan",
    "spectroscopy_scan",
]

