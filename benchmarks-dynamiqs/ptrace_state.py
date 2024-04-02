import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "ptrace_state"

samples = 5
evals = 100
cutoffs = range(2, 16)


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
    return dq.todm(psi)


def f(psi, N):
    return dq.ptrace(psi, [0,3], (N,N,2,2))


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi = setup(N)
    checks[N] = jnp.abs(f(psi, N)).sum()
    t = benchmarkutils.run_benchmark(f, psi, N, samples=samples, evals=evals)
    results.append({"N": 4 * N**2, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
