import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    alpha = jnp.log(N)
    return (N, alpha)

def f(N, alpha):
    return dq.displace(N, alpha)

def check_f(N, alpha):
    return float(jnp.real(dq.expect(dq.destroy(N), f(N, alpha))))

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