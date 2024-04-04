import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "timeevolution_master_cavity"

samples = 3
evals = 6
cutoffs = range(10, 151, 10)

def setup(N):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8)
    return solver

def f(N, solver):
    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tspan = jnp.linspace(0, 10, 11)

    a = dq.destroy(N)
    adag = dq.create(N)
    n = adag @ a

    H = delta_c*n + eta*(a + adag)
    J = jnp.sqrt(kappa) * a

    psi0 = dq.coherent(N, alpha0)
    
    exp_n = dq.mesolve(H, [J], psi0, tspan, exp_ops = [n], solver=solver).expects[0,:]
    return jnp.real(exp_n)


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    solver = setup(N)
    checks[N] = sum(f(N, solver))
    t = benchmarkutils.run_benchmark(f, N, solver, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)
