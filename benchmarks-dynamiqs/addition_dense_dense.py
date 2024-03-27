import jax.numpy as jnp
from jax import jit
import jax.random as random
import benchmarkutils

name = "addition_dense_dense"

samples = 2
evals = 100
cutoffs = range(50, 801, 50)
key = random.key(42)

def setup(N):
    op1 = random.uniform(key, [N, N]) * 0.2j
    op2 = random.uniform(key, [N, N]) * 0.1j
    return op1, op2

@jit
def f(op1, op2):
    return op1 + op2

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
