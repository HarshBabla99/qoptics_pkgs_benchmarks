import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "ptrace_operator"

samples = 5
evals = 100
cutoffs = range(2, 16)


def setup(N):
    def create_suboperator(c0, alpha, N):
        x = jnp.linspace(0., 1., N**2)
        return jnp.reshape(c0 + alpha * x , (N, N))

    op1 = create_suboperator(1, 0.2, N)
    op2 = create_suboperator(-2, 0.3, N)
    op3 = create_suboperator(3, 0.4, 2)
    op4 = create_suboperator(4, 0.5, 2)
    op = dq.tensor(op1, op2, op3, op4)
    return op


def f(op, N):
    return dq.ptrace(op, (0, 3), (N,N,2,2))


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    op = setup(N)
    checks[N] = jnp.abs(f(op, N)).sum()
    t = benchmarkutils.run_benchmark(f, op, samples=samples, evals=evals)
    results.append({"N": 4*N**2, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
