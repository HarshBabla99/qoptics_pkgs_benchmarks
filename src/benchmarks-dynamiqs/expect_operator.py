import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    op = (dq.destroy(N) + dq.create(N))
    rho = dq.todm(jnp.ones([N,1], complex))/N
    return (op, rho)

def f(op, rho):
    return dq.expect(op, rho)

if __name__ == '__main__':
    benchmark(name    = 'expect_operator', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(100, 2501, 100),
              check_f = f,
              to_jit  = False)