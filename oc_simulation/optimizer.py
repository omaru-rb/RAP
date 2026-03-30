import numpy as np 
from qutip import Qobj, basis, mesolve, sigmax, sigmaz, sigmay, expect, Result, Options, tensor, qeye
from qutip_qtrl import pulseoptim
from dataclasses import dataclass


@dataclass
class Optimizer:
    T : float
    dt : float
    omega : float
    Points : int
    Detuning : list
    Amplitude : list
    H : list | None = None
    Utarget : Qobj | None = None


    def __post_init__(self):
        if self.Utarget is None :
            self.Utarget=tensor([sigmax() for i in range(self.Points)])

    def BuildH(self):
        sy=sigmay()
        sx=sigmax()
        sz=sigmaz()
        I=qeye(2)
        H=[]
        n=self.Points
        D=np.linspace(self.Detuning[0],self.Detuning[1],n)
        A=np.linspace(self.Amplitude[0],self.Amplitude[1],n)

        H.append(sum(
            (A[i]) * tensor([sigmax() if j==i else I for j in range(n)])
            for i in range(n)
        ))
        H.append(sum(
            (A[i]) * tensor([sigmay() if j==i else I for j in range(n)])
            for i in range(n)
        ))
        H.append(sum(
            (D[i]/2) * tensor([sigmaz() if j==i else I for j in range(n)])
            for i in range(n)
        ))
        self.H = H    
    
    #def Fidelity(self,):
    #return(F)

    def run(self):
        self.BuildH()
        result=pulseoptim.opt_pulse_crab_unitary(
           H_d = self.H[2], H_c = [self.H[0], self.H[1]],
           U_0 = tensor([qeye(2) for i in range(self.Points)]), U_targ = self.Utarget,
           num_tslots = int(self.T/self.dt), evo_time = self.T,
           amp_lbound = -3*self.omega, amp_ubound = 3*self.omega, alg_params = {'num_coeffs': 25},
           fid_err_targ = 1e-4, max_iter = 10000, max_wall_time=300,
        )
        times=[]
        for i in range(int(self.T/self.dt)):
            times.append(self.dt*i)
        return times,result

