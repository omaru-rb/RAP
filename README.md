# Rapid Adiabatic Passage Simulation

A Python package for simulating rapid adiabatic passage (RAP) in atomic systems using QuTiP.

## Overview

This package provides tools for simulating and visualizing rapid adiabatic passage experiments, particularly for Rubidium-87 atoms. It uses QuTiP for quantum dynamics calculations and provides Bloch sphere visualizations.

## Features

- **Atom classes**: Extensible atomic species with physical constants (Rb-87 included)
- **Pulse shapes**: Constant, Gaussian, Lorentzian, Hyperbolic secant (extensible via registry)
- **Detuning profiles**: Linear, Tanh, Quadratic sweeps
- **Simulation**: Two-level system dynamics using QuTiP's `sesolve`
- **Analysis**: Adiabaticity criteria, Bloch trajectory analysis
- **Visualization**: Probability plots, Bloch sphere animations using QuTiP
- **Parameter sweeps**: Sweep time, frequency span, spectroscopy scans

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rapid_adiabatic_passage.git
cd rapid_adiabatic_passage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from rap_simulation import (
    Rubidium87,
    RapidAdiabaticPassage,
    SimulationParams,
)
from rap_simulation.visualization import plot_probabilities, plot_bloch_trajectory

# Create atom and simulation parameters
atom = Rubidium87()
params = SimulationParams.for_rb87_clock(
    T=1e-3,              # 1 ms total time
    rabi_freq_hz=10000,  # 10 kHz Rabi frequency
    freq_span_hz=40000,  # 40 kHz frequency span
    sweep_time=0.3e-3,   # 0.3 ms half-sweep time
)

# Run simulation
rap = RapidAdiabaticPassage(atom, params)
result = rap.run(pulse="hyper_secant", detuning="linear")

print(f"Transfer efficiency: {result.final_p1:.4f}")

# Visualize results
plot_probabilities(result)
plot_bloch_trajectory(result)
```

## Project Structure

```
rapid_adiabatic_passage/
├── rap_simulation/
│   ├── __init__.py
│   ├── atoms/
│   │   ├── base.py          # Atom base class
│   │   └── rubidium.py      # Rubidium-87
│   ├── pulses.py            # Pulse shape functions
│   ├── detuning.py          # Detuning sweep functions
│   ├── simulation.py        # RapidAdiabaticPassage class
│   ├── analysis.py          # Adiabaticity analysis
│   ├── scans.py             # Parameter sweep utilities
│   └── visualization/
│       ├── plots.py         # Matplotlib plots
│       └── bloch_animation.py  # Bloch sphere visualization
├── notebooks/
│   └── examples.ipynb       # Example notebook
├── requirements.txt
└── README.md
```

## Available Pulse Shapes

| Name | Description |
|------|-------------|
| `constant` | Rectangular pulse |
| `gaussian` | Gaussian envelope |
| `lorentzian` | Lorentzian envelope |
| `hyper_secant` | Hyperbolic secant (optimal for adiabatic passage) |

## Available Detuning Profiles

| Name | Description |
|------|-------------|
| `linear` | Linear frequency sweep |
| `tanh` | Hyperbolic tangent (S-curve) |
| `quadratic` | Quadratic sweep (slower near resonance) |

## Adding Custom Pulse Shapes

```python
from rap_simulation.pulses import register_pulse

@register_pulse("my_pulse")
def my_custom_pulse(t, t_center, omega, sweep_time, **kwargs):
    # Your pulse shape logic
    return omega * ...
```

## Dependencies

- QuTiP >= 5.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- tqdm >= 4.60.0

## License

MIT License

