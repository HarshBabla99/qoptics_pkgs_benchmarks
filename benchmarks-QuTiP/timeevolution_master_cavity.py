import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    options = qt.Options(atol = 1e-8, rtol = 1e-6)

    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tlist = np.linspace(0, 10, 11)

    a = qt.destroy(N)
    adag = qt.create(N)
    n = adag * a

    H = delta_c*n + eta*(a + adag)
    J = [np.sqrt(kappa) * a]

    psi0 = qt.coherent(N, alpha0)

    args = {
        'H'       : H, 
        'c_ops'   : [J], 
        'psi0'    : psi0, 
        'tlist'   : tlist, 
        'exp_ops' : [n], 
        'options'  : options
    }
    
    return (N,args)

def f(N,args):
    exp_n = qt.mesolve(args['H'], args['psi0'], args['tlist'], c_ops = args['c_ops'],
                       e_ops = args['exp_ops'], options=args['options']).expect[0]
    return np.real(exp_n)

def check_f(N,args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_master_cavity', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(10, 151, 10),
              check_f = check_f)
