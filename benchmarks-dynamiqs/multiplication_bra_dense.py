import dynamiqs as dq
import benchmarkutils
from jax import random
from jax import jit

name = "multiplication_bra_dense"

samples = 2
evals = 100
cutoffs = range(50, 501, 50)

key = random.PRNGKey(42)

def setup(N):
    op1 = dq.rand_dm(key, (N, N)) * 0.2j
    psi = dq.dag(dq.rand_ket(key, (N,1)))
    return op1, psi

@jit
def f(op1, psi):
    return psi @ op1

print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op1, psi = setup(N)
    t = benchmarkutils.run_benchmark(f, op1, psi,
                                     samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()
benchmarkutils.save(name, results)
