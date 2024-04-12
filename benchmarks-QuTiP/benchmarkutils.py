import timeit
import json

import qutip as qt
import numpy as np

from os.path import exists

benchmark_directory = "benchmarks-QuTiP"
commitID = qutip.version.version
result_path = "../results/results-QuTiP-{}-{}.json"

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
    check_path = "../checks/" + examplename(name) + ".json"
    
    if exists(check_path):
        print("\t Checking against check file.")

        f = open(check_path)
        data = json.load(f)
        for (N, result) in D.items():
            r = data[str(N)]
            if np.isnan(result) or abs(result-r)/abs(r) > eps:
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

def benchmark(name, f, setup, samples, evals, cutoffs, check_f, check_thresh=1e-5):

    print(f"Benchmarking: {name}", flush = True)
    print("Cutoff: ", end="", flush=True)
    
    results = []
    if check_f is not None:
        checks = {}

    # Now run this for various cutoffs
    for N in cutoffs:
        print(N, "", end="", flush=True)
        setup_args = setup(N)

        if check_f is not None:
            checks[N] = float(np.real(check_f(*setup_args)))

        t = run(f, *setup_args, 
                samples=samples, evals=evals)

        results.append({"N": N, "t": t})
    print()

    # Check the results
    if check_f is not None:
        check(name, checks, check_thresh)

    # Save the results to a file
    save(name, results)