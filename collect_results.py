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
    "timeevolution_schrodinger_cavity",
    "timeevolution_schrodinger_jaynescummings"
]

# This gives the "version" of each file i.e. deals with the variants in packages (QuTip, dynamiqs) & computational methods (cpu, gpu, etc.)
def extract_version(filename, testname):
    # First, let's deal with "(save_states)" or not. This is a seperate test, not a version
    if filename.endswith(")"):
        save_states = True
        name, _ = filename.rsplit("(", 1)
    else:
        save_states = False
        name = filename

    # Next, let's deal with the variants in computational methods 
    if name.endswith("]"):
        name, variant = name.rsplit("[", 1)
        variant = "/" + variant[:-1]
    else:
        variant = ""
    assert name.startswith("results-"), name
    assert name.endswith("-" + testname), name

    # Final "version" is in the form [package]-[version]/[method] e.g. dynamiqs-0.2.0/gpu
    version = name[len("results-"):-len("-" + testname)] + variant
    # version_escaped = version.replace(".", "_")

    # Return version & flag for save_states
    return version, save_states

def cutdigits(x):
    return float('%.3g' % (x))

def main():

    # Get a list of files in the "results" directory, and remove the file extension
    filenames = [os.path.splitext(file)[0] for file in os.listdir("results")]

    # Iterate through the tests listed above
    for testname in names:
        print("Loading: ", testname)

        # Get all the files which contain [testname] in their name 
        # (this should get the files for the different packages QuTiP, dynamiqs, etc. 
        # as well as the computational variants e.g. cpu, gpu, cython etc.)
        matches = filter(lambda x: testname in x, filenames)

        # dict to collate the data for all the matches i.e. version (key) vs. data (value)
        # two dicts, one for the save_states and one for the regular one. 
        # for all non time-evo tests, the save_states dict is unused
        d = {}
        d_save_states = {}
        for filename in matches:
            # Get the version and save_states flag
            version, save_states = extract_version(filename, testname)

            # Read the data
            with open(f"results/{filename}.json", 'r') as f:
                data = json.load(f)

            # Round the data (data is a key ("N"), value ("t") pair)
            for point in data:
                point["t"] = float('%.3g' % (point["t"]))

            # Add to the collated dict
            if save_states:
                d_save_states[version] = data
            else:
                d[version] = data

        # Save the collated results
        path = f"results-collected/{testname}.json"
        with open(path, 'w') as f:
            json.dump(d, f)
            print(f"Saved: {testname}")
        
        if d_save_states:
            path = f"results-collected/{testname}(save_states).json"
            with open(path, 'w') as f:
                json.dump(d_save_states, f)
                print(f"Saved: {testname}(save_states)")
        
        print()
        
if __name__ == '__main__':
    main()
