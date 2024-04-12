import qutip as qt
from benchmarkutils import benchmark

def setup(N):
    op1 = qt.rand_dm(N, N) * 0.2j
    psi = qt.rand_ket(N).dag().full().ravel()
    return (op1, psi)

def f(op1, psi):
    return psi * op1

if __name__ == '__main__':
    benchmark(name    = 'multiplication_bra_dense', 
              f       = f,
              setup   = setup,
              samples = 2,
              evals   = 100,
              cutoffs = range(50, 501, 50),
              check_f = None,
              to_jit  = True)
