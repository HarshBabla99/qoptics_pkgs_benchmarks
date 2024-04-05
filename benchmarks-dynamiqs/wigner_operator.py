import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    alpha = 0.7
    state = dq.coherent_dm(N, alpha)
    return (N,state)

def f(N,state):
    return dq.wigner(state, xmax = 50, ymax = 500, npixels = 100)

def check_f(N,state):
    alpha_check = 0.6 + 0.1j
    w = f(N,state)
    return w[alpha_check.real, alpha_check.imag]

if __name__ == '__main__':
    benchmark(name    = 'wigner_operator', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 5,
              cutoffs = range(10, 101, 10),
              check_f = None, #!!!!! check_f 
              to_jit  = False)
