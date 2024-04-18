import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    alpha = 0.7
    
    xvec = jnp.linspace(-5, 5, 100)
    yvec = jnp.linspace(-5, 5, 100)

    state = dq.coherent_dm(N, alpha)
    return (state, xvec, yvec)
 
def f(state, xvec, yvec):
    return dq.wigner(state, xvec = xvec, yvec = yvec)[2]

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
