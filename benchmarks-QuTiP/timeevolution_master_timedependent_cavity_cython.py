import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    options = qt.Options(atol = 1e-8, rtol = 1e-6, nsteps = 1000000)
    
    kappa = 1.
    eta = 1.5
    wc = 1.8
    wl = 2.
    # delta_c = wl - wc
    alpha0 = 0.3 - 0.5j
    tlist = np.linspace(0, 10, 11)

    a = qt.destroy(N)
    adag = qt.create(N)
    n = adag * a

    J = [np.sqrt(kappa) * a]
    psi0 = qt.coherent(N, alpha0)

    f1 = 'exp(1j*{wl}*t)'.format(wl=wl)
    f2 = 'exp(-1j*{wl}*t)'.format(wl=wl)

    H = [wc * n, [eta * a, f1], [eta * at, f2]]

    args = {
        'H'       : H, 
        'c_ops'   : [J], 
        'psi0'    : psi0, 
        'tlist'   : tlist, 
        'exp_ops' : [a], 
        'options'  : options
    }
    return (N,args)

def f(N, args):
    alpha_t = qt.mesolve(args['H'], args['psi0'], args['tlist'], c_ops = args['c_ops'],
                         e_ops = args['exp_ops'], options=args['options']).expect[0]

    return np.real(alpha_t)

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
              to_jit  = False)