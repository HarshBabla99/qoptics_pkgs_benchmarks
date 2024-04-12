import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    def create_substate(c0, alpha, N):
        x = np.linspace(0., 1., N).reshape([N,1])
        data = (c0 + alpha * x).conj()
        return qt.Qobj(data).unit()

    psi1 = create_substate(1, 0.2, N)
    psi2 = create_substate(-2, 0.3, N)
    psi3 = create_substate(3, 0.4, 2)
    psi4 = create_substate(4, 0.5, 2)
    psi = dq.tensor(psi1, psi2, psi3, psi4)

    # Note the corresponding dynamiqs code had to convert this to a dm first
    # no need to do that for qutip
    return (N, psi)

def f(N, psi):
     return qt.ptrace(psi, [0, 3])

def check_f(N, psi):
    return np.abs(f(N,psi)).sum()

if __name__ == '__main__':
    benchmark(name    = 'ptrace_state', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(2, 16),
              check_f = check_f,
              to_jit  = False)