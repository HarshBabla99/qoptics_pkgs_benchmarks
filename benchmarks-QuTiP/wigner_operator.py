import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    alpha = 0.7
    
    xvec = np.linspace(-5, 5, 100)
    yvec = np.linspace(-5, 5, 100)

    state = qt.coherent_dm(N, alpha)
    return (state, xvec, yvec)
 
def f(state, xvec, yvec):
    return qt.wigner(state, xvec = xvec, yvec = yvec, g= 2.0)

def check_f(state, xvec, yvec):
    alpha_check = 0.6 + 0.1j
    return f(state, [alpha_check.real], [alpha_check.imag])[0,0]

if __name__ == '__main__':
    benchmark(name    = 'wigner_operator', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 5,
              cutoffs = range(10, 101, 10),
              check_f = check_f,
              to_jit  = False)
