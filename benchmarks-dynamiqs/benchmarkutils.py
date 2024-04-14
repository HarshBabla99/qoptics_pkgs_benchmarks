import timeit
import json

import dynamiqs as dq

import jax.numpy as jnp
from jax import jit
from jax.lib import xla_bridge
from jax import config as jax_config

from os.path import exists

benchmark_directory = "benchmarks-dynamiqs"
commitID = dq.__version__
result_path = "../results/results-dynamiqs-{}-{}.json"

def examplename(name):
    if name.endswith("]"):
        return name.rsplit("[", 1)[0]
    else:
        return name

def run(f, *args, samples=5, evals=1):
    dq.set_precision('double')

    D = {"f": f, "args": args}
    t = timeit.repeat("f(*args)", globals=D, number=evals, repeat=samples)
    return min(t)/evals

def check(name, D, eps=1e-5):
    check_path = "../checks/" + examplename(name) + ".json"
    
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

def benchmark(name, f, setup, samples, evals, cutoffs, check_f, check_thresh=1e-5, to_jit=False):

    # Get the backends
    if xla_bridge.get_backend().platform == 'gpu':
        backends = ['cpu', 'gpu']
    else:
        backends = ['cpu']

    # Run over jit/no jit if specified 
    if to_jit:
        jits = [False, True]
    else:
        jits = [False]

    for jit_status in jits:
        for backend in backends:

            print(f"Benchmarking:{name} - {backend} - jit = {jit_status}")
            print("Cutoff: ", end="", flush=True)
            
            results = []
            if check_f is not None:
                checks = {}

            # Set the backend
            dq.set_device(backend)

            # To jit or not to jit, that is the question
            if jit_status:
                f_to_run = jit(f, backend = backend)
                curr_name = f'{name}[{backend}_jit]'

            else:
                f_to_run = f
                curr_name = f'{name}[{backend}]'

            # Now run this for various cutoffs
            for N in cutoffs:
                print(N, "", end="", flush=True)
                setup_args = setup(N)

                # If the function is "jitted" then run it once
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