from rap_simulation import Rubidium87, HyperfineState, Transition, SimulationParams, RapidAdiabaticPassage
from rap_simulation.visualization.plots import plot_pulse_and_detuning, plot_probabilities
import numpy as np


def main():
    stretch_transition = Rubidium87.list_transitions()[1]
    b_field = 8e-4  #tesla
    atom = Rubidium87(B_field=b_field, transition=stretch_transition)

    print(atom)
    print(atom.transition)
    print(atom.transition_frequency)

    parameters = SimulationParams(
        T = 1e-3,
        dt = 0.1e-6,
        omega = 10e3,
        sweep_time = 1e-3,
        freq_span = 10e3,
        freq_span_center = 0.0,
    )

    rap = RapidAdiabaticPassage(atom, parameters)
    result = rap.run(pulse="gaussian")
    
    fig = plot_pulse_and_detuning(result, time_unit="ms", freq_unit="kHz")
    fig2 = plot_probabilities(result, time_unit="ms")
    fig.savefig("pulse_and_detuning.png")
    fig2.savefig("probabilities.png")


if __name__ == "__main__":
    main()