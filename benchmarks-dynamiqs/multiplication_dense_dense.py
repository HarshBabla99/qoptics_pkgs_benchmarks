import jax.numpy as jnp
import benchmarkutils

name = "multiplication_dense_dense"

samples = 2
evals = 100
cutoffs = range(50, 601, 50)

key = jax.random.PRNGKey(42)

def setup(N):
    op1 = dq.rand_dm(key, (N, N)) * 0.2j
    op2 = dq.rand_dm(key, (N, N)) * 0.1j
    return op1, op2

def f(op1, op2):
    return op1 @ op2

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op1, op2 = setup(N)
    t = benchmarkutils.run_benchmark(f, op1, op2, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.save(name, results)
