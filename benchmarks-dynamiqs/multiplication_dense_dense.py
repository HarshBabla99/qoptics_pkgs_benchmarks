import dynamiqs as dq
from jax import random
from benchmarkutils import benchmark

def setup(N):
    key = random.PRNGKey(42)
    op1 = dq.rand_dm(key, (N, N)) * 0.2j
    op2 = dq.rand_dm(key, (N, N)) * 0.1j
    return (op1, op2)

def f(op1, op2):
    return op1 @ op2

if __name__ == '__main__':
    benchmark(name    = 'multiplication_dense_dense', 
              f       = f,
              setup   = setup,
              samples = 2,
              evals   = 100,
              cutoffs = range(50, 601, 50),
              check_f = None,
              to_jit  = True)

