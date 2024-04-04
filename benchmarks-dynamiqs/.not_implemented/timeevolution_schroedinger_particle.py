import dynamiqs as dq
import jax.numpy as jnp
import benchmarkutils

name = "timeevolution_schroedinger_particle"

samples = 3
evals = 6
cutoffs = range(25, 301, 25)


def setup(N):
    xmin = -5
    xmax = 5
    x0 = 0.3
    p0 = -0.2
    sigma0 = 1
    dx = (xmax - xmin) / N
    pmin = -jnp.pi / dx
    pmax = jnp.pi / dx
    dp = (pmax - pmin) / N

    samplepoints_x = jnp.linspace(xmin, xmax, N, endpoint=False)
    samplepoints_p = jnp.linspace(pmin, pmax, N, endpoint=False)

    x = jnp.diag(samplepoints_x)
    row0 = [sum([p*jnp.exp(-1j*p*dxji) for p in samplepoints_p])*dp*dx/(2*jnp.pi) 
                for dxji in samplepoints_x - xmin]
    row0 = jnp.reshape(jnp.array(row0), [len(row0), 1])
    col0 = row0.conj()

    p = jnp.zeros([N, N], dtype=complex)
    for i in range(N):
        p[i, i:] = row0[:N - i]
        p[i:, i] = col0[:N - i]

    H = dq.powm(p,2) + 2 * dq.powm(x,2)

    def gaussianstate(x0, p0, sigma0):
        alpha = 1./(jnp.pi**(1/4)*jnp.sqrt(sigma0))*jnp.sqrt(dx)
        data = alpha*jnp.exp(1j*p0*(samplepoints_x-x0/2) - dq.powm(samplepoints_x-x0, 2)/(2*sigma0**2))
        return jnp.reshape(data,[len(data),1])

    psi0 = gaussianstate(x0, p0, sigma0)

    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8, nsteps = 1000000)

    return psi0, H, x, solver


def f(psi0, H, x, options):
    tlist = jnp.linspace(0, 10, 11)
    exp_x = dq.sesolve(H, psi0, tlist, exp_ops = [x], solver=solver).expects[0,:]
    return exp_x


print("Benchmarking:", name)
print("Cutoff: ", end="", flush=True)
checks = {}
results = []
for N in cutoffs:
    print(N, "", end="", flush=True)
    psi0, H, x, solver = setup(N)
    checks[N] = sum(f(psi0, H, x, solver))
    t = benchmarkutils.run_benchmark(f, psi0, H, x, solver, samples=samples, evals=evals)
    results.append({"N": N, "t": t})
print()

benchmarkutils.check(name, checks, 1e-4)
benchmarkutils.save(name, results)
