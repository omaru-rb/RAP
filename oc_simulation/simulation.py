import numpy as np 
from qutip import Qobj, basis, mesolve, sigmax, sigmaz, sigmay, expect, Result, Options, tensor, qeye
from qutip_qtrl import pulseoptim
from dataclasses import dataclass

@dataclass
class OCSimulation:
    T : float
    dt : float
    Times : list
    Amps : list
    Omega : float
    Detuning : float
    Offset : float
    H : list | None = None

    def BuildH(self):
        D=self.Detuning
        Off=self.Offset
        sy=sigmay()
        sx=sigmax()
        sz=sigmaz()
        I=qeye(2)
        Hz=[(D/2)*sz,np.array([1 for i in range(len(self.Amps))], dtype=np.float64)]
        Hx=[sx*Off, np.array([A[0] for A in self.Amps], dtype=np.float64)]
        Hy=[sy*Off, np.array([A[1] for A in self.Amps], dtype=np.float64)]
        self.H=[Hz,Hx,Hy]

    def run(self):
        psi0=basis(2,0)
        psi1=basis(2,1)
        self.BuildH()
        opts = Options(max_step=1e-6)
        output=mesolve(self.H, psi0, self.Times, options=opts)

        # Extract state amplitudes
        c0 = np.array([psi0.dag() * state for state in output.states])
        c1 = np.array([psi1.dag() * state for state in output.states])
        
        # Probabilities
        p0 = np.abs(c0)**2
        p1 = np.abs(c1)**2
        return output,p1
