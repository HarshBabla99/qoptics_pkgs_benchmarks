import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    alpha = np.log(N)
    return (N, float(alpha))

def f(N, alpha):
    return qt.coherent(N, alpha, method="analytic")

def check_f(N, alpha):
    return np.real(qt.expect(qt.destroy(N), f(N, alpha)))

if __name__ == '__main__':
    benchmark(name    = 'coherentstate', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(50, 501, 50),
              check_f = check_f,
              check_thresh = 0.05)