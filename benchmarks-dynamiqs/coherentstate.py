import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    alpha = jnp.log(N)
    return (N, float(alpha))

def f(N, alpha):
    return dq.coherent(N, alpha)

def check_f(N, alpha):
    return jnp.real(dq.expect(dq.destroy(N), f(N, alpha)))

if __name__ == '__main__':
    benchmark(name    = 'coherentstate', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(50, 501, 50),
              check_f = check_f,
              check_thresh = 0.05,
              to_jit  = False)