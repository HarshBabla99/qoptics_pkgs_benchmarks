import dynamiqs as dq
from jax import random
from benchmarkutils import benchmark

def setup(N):
    key = random.PRNGKey(42)
    op1 = dq.rand_dm(key, (N, N)) * 0.2j
    psi = dq.dag(dq.rand_ket(key, (N,1)))
    return (op1, psi)

def f(op1, psi):
    return psi @ op1

if __name__ == '__main__':
    benchmark(name    = 'multiplication_bra_dense', 
              f       = f,
              setup   = setup,
              samples = 2,
              evals   = 100,
              cutoffs = range(50, 501, 50),
              check_f = None,
              to_jit  = True)
