import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    def create_substate(c0, alpha, N):
        x = jnp.linspace(0., 1., N).reshape([N,1])
        data = (c0 + alpha * x).conj()
        return dq.unit(data)

    psi1 = create_substate(1, 0.2, N)
    psi2 = create_substate(-2, 0.3, N)
    psi3 = create_substate(3, 0.4, 2)
    psi4 = create_substate(4, 0.5, 2)
    psi = dq.tensor(psi1, psi2, psi3, psi4)

    # Note I convert this to a dm first. 
    # When I directly send the ket, it fails (with the error x2APIC is not supported)
    return (N, dq.todm(psi))

def f(N, psi):
    return dq.ptrace(psi, [0,3], (N,N,2,2))

def check_f(N, psi):
    return jnp.abs(f(N,psi)).sum()

if __name__ == '__main__':
    benchmark(name    = 'ptrace_state', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(2, 16),
              check_f = check_f,
              to_jit  = False)