import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "timeevolution_master_timedependent_jaynescummings"

samples = 3
evals = 6
cutoffs = range(5, 81, 5)

def setup(N):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8)
    return solver

def f(N, solver):
    wa = 1
    wc = 0.9
    delta = wa - wc
    g = 2
    kappa = 0.5
    gamma = 0.1
    n_th = 0.75
    tspan = jnp.linspace(0, 10, 11)

    Ia = dq.eye(2)
    Ic = dq.eye(N)

    a = dq.destroy(N)
    adag = dq.create(N)

    sm = dq.sigmam()
    sp = dq.sigmap()

    H1 = g * dq.tensor(adag, sm)
    H2 = g * dq.tensor(a   , sp)

    f1 = lambda t: jnp.exp(-1j*delta*t)
    f2 = lambda t: jnp.exp(1j*delta*t)

    H = dq.modulated(f1, H1) + dq.modulated(f2, H2)

    c_ops = [
        dq.tensor(jnp.sqrt(kappa*(1+n_th)) * a   , Ia),
        dq.tensor(jnp.sqrt(kappa*n_th)     * adag, Ia),
        dq.tensor(Ic                             , jnp.sqrt(gamma) * sm),
    ]

    psi0 = dq.tensor(dq.fock(N, 0), dq.unit(dq.basis(2, 0) + dq.basis(2, 1)))

    exp_n = dq.mesolve(H, c_ops, psi0, tspan, exp_ops = [dq.tensor(a, sp)],
                       solver=solver).expects[0,:]

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
    results.append({"N": 2*N, "t": t})
print()

benchmarkutils.check(name, checks)
benchmarkutils.save(name, results)