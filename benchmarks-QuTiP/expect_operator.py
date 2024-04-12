import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    op = (qt.destroy(N) + qt.create(N))
    rho = qt.todm(np.ones([N,1], complex))/N
    return (op, rho)

def f(op, rho):
    return qt.expect(op, rho)

if __name__ == '__main__':
    benchmark(name    = 'expect_operator', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(100, 2501, 100),
              check_f = f,
              to_jit  = False)