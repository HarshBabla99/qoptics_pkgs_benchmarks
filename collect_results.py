import os
import json

names = [
    "addition_dense_dense",
    "multiplication_dense_dense",
    "multiplication_dense_ket",
    "multiplication_bra_dense",
    "coherentstate",
    "displace",
    "expect_operator",
    "expect_state",
    "ptrace_operator",
    "ptrace_state",
    "wigner_operator",
    "wigner_state",
    "timeevolution_master_cavity",
    "timeevolution_master_jaynescummings",
    "timeevolution_master_timedependent_cavity",
    "timeevolution_master_timedependent_jaynescummings",
    "timeevolution_schroedinger_cavity",
    "timeevolution_schroedinger_jaynescummings"
]

filenames = os.listdir("results")

def extract_version(filename, testname):
    name, _ = os.path.splitext(filename)
    if name.endswith("]"): # For example timeevolution_particle[fft]
        name, variant = name.rsplit("[", 1)
        variant = "/" + variant[:-1]
    else:
        variant = ""
    assert name.startswith("results-"), name
    assert name.endswith("-" + testname), name
    return name[len("results-"):-len("-" + testname)] + variant

def cutdigits(x):
    return float('%.3g' % (x))

for testname in names:
    print("Loading: ", testname)
    matches = filter(lambda x: testname in x, filenames)
    d = {}
    for filename in matches:
        version = extract_version(filename, testname)
        f = open("results/" + filename)
        # version_escaped = version.replace(".", "_")
        data = json.load(f)
        for point in data:
            point["t"] = cutdigits(point["t"])
        d[version] = data
        f.close()
    path = "results-collected/" + testname + ".json"
    f = open(path, "w")
    json.dump(d, f)
    f.close()
