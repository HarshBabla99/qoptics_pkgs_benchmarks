import qutip as qt
import numpy as np
from benchmarkutils import benchmark

def setup(N):
    def create_suboperator(c0, alpha, N):
        x = np.linspace(0., 1., N**2)
        return qt.Qobj(np.reshape(c0 + alpha * x , (N, N)))

    op1 = create_suboperator(1, 0.2, N)
    op2 = create_suboperator(-2, 0.3, N)
    op3 = create_suboperator(3, 0.4, 2)
    op4 = create_suboperator(4, 0.5, 2)
    op = qt.tensor(op1, op2, op3, op4)
    return (N,op)

def f(N,op):
    return qt.ptrace(op, (0, 3))

def check_f(N, op):
    return np.abs(f(N,op).full()).sum()

if __name__ == '__main__':
    benchmark(name    = 'ptrace_operator', 
              f       = f,
              setup   = setup,
              samples = 5,
              evals   = 100,
              cutoffs = range(2, 16),
              check_f = check_f)
