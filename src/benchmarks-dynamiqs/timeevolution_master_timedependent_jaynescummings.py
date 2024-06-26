import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N, save_states):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8, max_steps = 1000000)
    options = dq.Options(save_states = save_states)

    wa = 1
    wc = 0.9
    delta = wa - wc
    g = 2
    kappa = 0.5
    gamma = 0.1
    n_th = 0.75
    tlist = jnp.linspace(0, 10, 11)

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

    args = {
            'H'       : H, 
            'c_ops'   : c_ops, 
            'psi0'    : psi0, 
            'tlist'   : tlist, 
            'exp_ops' : [dq.tensor(a, sp)], 
            'solver'  : solver,
            'options' : options
        }
    
    return (N,args)

def f(N, args):
    exp_n = dq.mesolve(args['H'], args['c_ops'], args['psi0'], args['tlist'],
                       exp_ops = args['exp_ops'], 
                       solver=args['solver'], options = args['options']).expects[0,:]

    return jnp.real(exp_n)

def check_f(N, args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_master_timedependent_jaynescummings', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(5, 81, 5),
              check_f = check_f,
              to_jit  = False,
              is_time_evo = True)