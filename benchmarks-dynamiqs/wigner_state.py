import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "wigner_state"

samples = 3
evals = 5
cutoffs = range(10, 101, 10)

def setup(N):
    alpha = 0.7
    xvec = jnp.linspace(-50, 50, 100)
    yvec = jnp.linspace(-50, 50, 100)
    state = dq.coherent(N, alpha)
    return state, xvec, yvec

def f(state, xvec, yvec):
    return dq.wigner(state, xvec, yvec)

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    state, xvec, yvec = setup(N)
    alpha_check = 0.6 + 0.1j
    checks[N] = f(state, [alpha_check.real], [alpha_check.imag])[0, 0]
    t = benchmarkutils.run_benchmark(f, state, xvec, yvec, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
