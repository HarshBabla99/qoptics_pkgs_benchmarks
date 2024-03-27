import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "expect_state"

samples = 5
evals = 100
cutoffs = range(5000, 150001, 5000)

def setup(N):
    op = (dq.destroy(N) + dq.create(N))
    psi = jnp.ones(N, complex)/jnp.sqrt(N)
    return op, psi

def f(op, psi):
    return dq.expect(op, psi)

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op, psi = setup(N)
    checks[N] = f(op, psi)
    t = benchmarkutils.run_benchmark(f, op, psi, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
