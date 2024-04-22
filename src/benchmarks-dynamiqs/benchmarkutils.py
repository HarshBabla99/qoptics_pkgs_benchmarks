import timeit
import json

import dynamiqs as dq

import jax.numpy as jnp
from jax import jit
from jax.lib import xla_bridge
from jax import config as jax_config

from os.path import exists
import itertools

benchmark_directory = "benchmarks-dynamiqs"
commitID = dq.__version__
result_path = "../../out/results/results-dynamiqs-{}-{}.json"

def examplename(name):
    if name.endswith("]"):
        return name.rsplit("[", 1)[0]
    else:
        return name

def run(f, *args, samples=5, evals=1):
    D = {"f": f, "args": args}
    t = timeit.repeat("f(*args)", globals=D, number=evals, repeat=samples)
    return min(t)/evals

def check(name, D, eps=1e-5):
    check_path = "../../out/checks/" + examplename(name) + ".json"
    
    if exists(check_path):
        print("\t Checking against check file.")

        f = open(check_path)
        data = json.load(f)
        for (N, result) in D.items():
            r = data[str(N)]
            if jnp.isnan(result) or abs(result-r)/abs(r) > eps:
                print("Warning: Result may be incorrect in", name, ": ", result, "<->", r)

    else:
        print("\t No check file found - write results to check file.")
        f = open(check_path, "w")
        json.dump(D, f)
        f.close()

def save(name, results):
    f = open(result_path.format(commitID, name), "w")
    json.dump(results, f)
    f.close()

def benchmark(name, f, setup, samples, evals, cutoffs, check_f, check_thresh=1e-5, 
              to_jit=False, is_time_evo = False):
    
    # Double precision
    # dq.set_precision('double')

    # Get the backends
    if xla_bridge.get_backend().platform == 'gpu':
        backend_options = ['cpu', 'gpu']
    else:
        backend_options = ['cpu']

    # Run over jit/no jit if specified 
    if to_jit:
        jit_options = [False, True]
    else:
        jit_options = [False]

    # This is a flag to for a time-evolution simulation or not
    # For time-evolutions, we iterate over the save-states, else we do not
    if is_time_evo:
        save_options = [False, True]
    else:
        save_options = [None]

    # Itearate over all variants
    iterable = itertools.product(backend_options, jit_options, save_options)
    for (backend, jit_status, save_states) in iterable:

        print(f"Benchmarking:{name}")
        print(f"\t> backend     = {backend}")
        print(f"\t> jit         = {jit_status}")
        if save_states is not None:
            print(f"\t> save states = {save_states}")
        print("\tCutoff: ", end="", flush=True)
        
        results = []
        if check_f is not None:
            checks = {}

        # Set the backend
        dq.set_device(backend)

        # Dummy variable for the name (will be modified by the backend, jit, etc.)
        curr_name = name

        # To jit or not to jit, that is the question
        if jit_status:
            f_to_run = jit(f, backend = backend)
            curr_name = f'{curr_name}[{backend}_jit]'

        else:
            f_to_run = f
            curr_name = f'{curr_name}[{backend}]'

        # If we are saving the states, then save the results to a different file
        if save_states is not None:
            if save_states:
                curr_name = f'{curr_name}(save_states)'
            else:
                curr_name = name
        else:
            curr_name = name

        # Now run this for various cutoffs
        for N in cutoffs:
            print(N, "", end="", flush=True)

            # The setup needs the save_states flag, but only if we're running a time evo. sim
            if save_states is None:
                setup_args = setup(N)
            else:
                setup_args = setup(N, save_states)

            # If the function is "jitted" then run it once (for compilation)
            if jit_status:
                _ = f_to_run(*setup_args)

            if check_f is not None:
                checks[N] = float(jnp.real(check_f(*setup_args)))

            t = run(f_to_run, *setup_args, 
                    samples=samples, evals=evals)

            results.append({"N": N, "t": t})
        print()

        # Check the results
        if check_f is not None:
            check(curr_name, checks, 0.05)

        # Save the results to a file
        save(curr_name, results)