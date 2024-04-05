import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8)

    wa = 1
    wc = 1
    g = 2
    tlist = jnp.linspace(0, 10, 11)

    Ia = dq.eye(2)
    Ic = dq.eye(N)

    a = dq.destroy(N)
    adag = dq.create(N)
    n = adag @ a

    sm = dq.sigmam()
    sp = dq.sigmap()
    sz = dq.sigmaz()

    H = wc*dq.tensor(n, Ia) + dq.tensor(Ic, 0.5*wa*sz) + g*(dq.tensor(adag, sm) + dq.tensor(a, sp))

    psi0 = dq.tensor(dq.fock(N, 0), dq.unit(dq.basis(2, 0) + dq.basis(2, 1)))

    args = {
            'H'       : H, 
            'psi0'    : psi0, 
            'tlist'   : tlist, 
            'exp_ops' : [dq.tensor(n, Ia)], 
            'solver'  : solver
        }
    
    return (N,args)

def f(N, args):
    exp_n = dq.sesolve(args['H'], args['psi0'], args['tlist'],
                       exp_ops = args['exp_ops'], solver=args['solver']).expects[0,:]

    return jnp.real(exp_n)

def check_f(N, args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_schroedinger_jaynescummings', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(25, 251, 25),
              check_f = check_f,
              to_jit  = False)
