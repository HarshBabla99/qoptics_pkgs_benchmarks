# 0. Code
Link to [Github](https://github.com/HarshBabla99/qoptics_pkgs_benchmarks)

# 1. Time Evolution
Benchmarks for `sesolve` (for the lossless case) and `mesolve` (for the lossy case)

### 1.1.1: Driven cavity (Schrodinger Evolution)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_schrodinger_cavity(save_states).json.png) | ![](plots/timeevolution_schrodinger_cavity.json.png) |

### 1.1.2: Driven & lossy cavity (Time-independent Master Equation)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_master_cavity(save_states).json.png) | ![](plots/timeevolution_master_cavity.json.png) |

### 1.1.3: Driven & lossy cavity (Time-dependent Master Equation)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_master_timedependent_cavity(save_states).json.png) | ![](plots/timeevolution_master_timedependent_cavity.json.png) |

### 1.2.1: Jaynes-Cummings Hamiltonian (Schrodinger Evolution)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_schrodinger_jaynescummings(save_states).json.png) | ![](plots/timeevolution_schrodinger_jaynescummings.json.png) |

### 1.2.2: Jaynes-Cummings Hamiltonian (Time-independent Master Equation)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_master_jaynescummings(save_states).json.png) | ![](plots/timeevolution_master_jaynescummings.json.png) |

### 1.2.3: Jaynes-Cummings Hamiltonian (Time-dependent Master Equation)

| Save intermediate states | Don't save intermediate states |
|----------|----------|
| ![](plots/timeevolution_master_timedependent_jaynescummings(save_states).json.png) | ![](plots/timeevolution_master_timedependent_jaynescummings.json.png) |


# 2. Helper functions

### 2.1: Expectation values

| On states | On operators |
|----------|----------|
| ![](plots/expect_state.json.png) | ![](plots/expect_operator.json.png) |

### 2.2: Wigner function

| On states | On operators |
|----------|----------|
| ![](plots/wigner_state.json.png) | ![](plots/wigner_operator.json.png) |

### 2.3: Partial trace

| On states | On operators |
|----------|----------|
| ![](plots/ptrace_operator.json.png) | ![](plots/ptrace_state.json.png) |

### 2.4: Coherent stes

| Using `coherent()` | Using `displace()` |
|----------|----------|
| ![](plots/coherentstate.json.png) | ![](plots/displace.json.png) |

# 3. Arithmetic
These are all on dense operators & vectors (since dynamiqs doesn't support sparse operators yet)

### 3.1 Addition
| Operator-operator | 
|----------|
| ![](plots/addition_dense_dense.json.png) |

### 3.2 Multiplication
| Operator-operator | Bra-operator | Operator-Ket |  
|----------|----------|----------| 
| ![](plots/multiplication_dense_dense.json.png) | ![](plots/multiplication_bra_dense.json.png) | ![](plots/multiplication_dense_ket.json.png) |  