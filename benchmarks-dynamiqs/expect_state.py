import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    op = (dq.destroy(N) + dq.create(N))
    psi = jnp.ones([N,1], complex)/jnp.sqrt(N)
    return (op, psi)

def f(op, psi):
    return dq.expect(op, psi)

if __name__ == '__main__':
    benchmark(name    = 'expect_state', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(500, 9001, 500),
              check_f = f,
              to_jit  = False)