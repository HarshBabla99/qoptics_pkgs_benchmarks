import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "wigner_operator"

samples = 3
evals = 5
cutoffs = range(10, 101, 10)

def setup(N):
    alpha = 0.7
    op = dq.coherent_dm(N, alpha)
    return op

def f(state):
    return dq.wigner(state, xmax = 50, ymax = 500, npixels = 100)

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)

# checks = {}

results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op = setup(N)
    
    # alpha_check = 0.6 + 0.1j
    # w = f(state)
    # checks[N] = w[alpha_check.real, alpha_check.imag]

    t = benchmarkutils.run_benchmark(f, op, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

# benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
