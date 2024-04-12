import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    alpha = np.log(N)
    return (N, alpha)

def f(N, alpha):
    return qt.displace(N, alpha)

def check_f(N, alpha):
    return float(np.real(qt.expect(qt.destroy(N), f(N, alpha))))

if __name__ == '__main__':
    benchmark(name    = 'displace', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 100,
              cutoffs = range(10, 151, 10),
              check_f = check_f,
              check_thresh = 0.05,
              to_jit  = False)