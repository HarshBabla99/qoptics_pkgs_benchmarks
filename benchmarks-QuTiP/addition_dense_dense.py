import numpy as np
from benchmarkutils import benchmark

def setup(N):
    op1 = np.random.rand(N, N) * 0.2j
    op2 = np.random.rand(N, N) * 0.1j
    return (op1, op2)

def f(op1, op2):
    return op1 + op2
  
if __name__ == '__main__':
    benchmark(name    = 'addition_dense_dense', 
              f       = f,
              setup   = setup,
              samples = 2,
              evals   = 100,
              cutoffs = range(50, 801, 50),
              check_f = None, 
              check_thresh = None)
    