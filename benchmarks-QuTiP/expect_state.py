import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    op = (qt.destroy(N) + qt.create(N))
    psi = qt.Qobj(np.ones(N, complex)/np.sqrt(N))
    return (op, psi)

def f(op, psi):
    return qt.expect(op, psi)

if __name__ == '__main__':
    benchmark(name    = 'expect_state', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(500, 10001, 500),
              check_f = f)