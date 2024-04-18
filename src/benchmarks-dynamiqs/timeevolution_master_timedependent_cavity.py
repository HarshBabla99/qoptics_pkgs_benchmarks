import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N, save_states):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8, max_steps = 1000000)
    options = dq.Options(save_states = save_states)

    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    # delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tlist = jnp.linspace(0, 10, 11)

    a = dq.destroy(N)
    adag = dq.create(N)
    n = adag @ a

    J = [jnp.sqrt(kappa) * a]
    psi0 = dq.coherent(N, alpha0)

    f1 = lambda t: jnp.exp(1j * wl * t)
    f2 = lambda t: jnp.exp(-1j * wl * t)

    H = dq.constant(wc * n) + dq.modulated(f1, eta * a) + dq.modulated(f2, eta * adag)

    args = {
        'H'       : H, 
        'c_ops'   : [J], 
        'psi0'    : psi0, 
        'tlist'   : tlist, 
        'exp_ops' : [a], 
        'solver'  : solver,
        'options' : options
    }
    return (N,args)

def f(N, args):
    alpha_t = dq.mesolve(args['H'], args['c_ops'], args['psi0'], args['tlist'],
                         exp_ops = args['exp_ops'], 
                         solver=args['solver'], options = args['options']).expects[0,:][0]

    return jnp.real(alpha_t)

def check_f(N,args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_master_timedependent_cavity', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(10, 141, 10),
              check_f = check_f,
              check_thresh = 1e-4,
              to_jit  = False,
              is_time_evo = True)