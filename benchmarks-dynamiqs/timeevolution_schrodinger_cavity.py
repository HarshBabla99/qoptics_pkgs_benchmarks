import dynamiqs as dq
import jax.numpy as jnp
from benchmarkutils import benchmark

def setup(N):
    solver = dq.solver.Tsit5(rtol = 1e-6, atol = 1e-8)
    
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tlist = jnp.linspace(0, 10, 11)

    a = dq.destroy(N)
    adag = dq.create(N)
    n = adag @ a

    H = delta_c*n + eta*(a + adag)
    psi0 = dq.coherent(N, alpha0)

    args = {
        'H'       : H, 
        'psi0'    : psi0, 
        'tlist'   : tlist, 
        'exp_ops' : [n], 
        'solver'  : solver
    }
    
    return (N,args)

def f(N, args):
    exp_n = dq.sesolve(args['H'], args['psi0'], args['tlist'],
                       exp_ops = args['exp_ops'], solver=args['solver']).expects[0,:]

    return jnp.real(exp_n)

def check_f(N,args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_schrodinger_cavity', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(50, 501, 50),
              check_f = check_f,
              to_jit  = False)