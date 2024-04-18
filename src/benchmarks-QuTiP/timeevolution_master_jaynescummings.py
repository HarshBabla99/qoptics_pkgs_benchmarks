import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N, save_states):
    options = qt.Options(atol = 1e-8, rtol = 1e-6, store_states = save_states)

    wa = 1
    wc = 1
    g = 2
    kappa = 0.5
    gamma = 0.1
    n_th = 0.75
    tlist = np.linspace(0, 10, 11)

    Ia = qt.qeye(2)
    Ic = qt.qeye(N)

    a = qt.destroy(N)
    adag = qt.create(N)
    n = adag * a

    sm = qt.sigmam()
    sp = qt.sigmap()
    sz = qt.sigmaz()

    H = wc*qt.tensor(n, Ia) + qt.tensor(Ic, 0.5*wa*sz) + g*(qt.tensor(adag, sm) + qt.tensor(a, sp))

    c_ops = [
        qt.tensor(np.sqrt(kappa*(1+n_th)) * a   , Ia),
        qt.tensor(np.sqrt(kappa*n_th)     * adag, Ia),
        qt.tensor(Ic                             ,np.sqrt(gamma) * sm),
    ]

    psi0 = qt.tensor(qt.fock(N, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())

    args = {
            'H'       : H, 
            'c_ops'   : c_ops, 
            'psi0'    : psi0, 
            'tlist'   : tlist, 
            'exp_ops' : [qt.tensor(n, Ia)], 
            'options'  : options
        }
    
    return (N,args)

def f(N, args):
    exp_n = qt.mesolve(args['H'], args['psi0'], args['tlist'], c_ops = args['c_ops'],
                       e_ops = args['exp_ops'], options=args['options']).expect[0]

    return np.real(exp_n)

def check_f(N, args):
    return sum(f(N,args))

if __name__ == '__main__':
    benchmark(name    = 'timeevolution_master_jaynescummings', 
              f       = f,
              setup   = setup,
              samples = 3,
              evals   = 6,
              cutoffs = range(5, 81, 5),
              check_f = check_f, 
              is_time_evo = True)