import os
import subprocess
import warnings

matlabbenchmarks = "benchmarks-QuantumOpticsToolbox"
juliabenchmarks = "benchmarks-QuantumOptics.jl"
qutipbenchmarks = "benchmarks-QuTiP"
dynamiqsbenchmarks = "benchmarks-dynamiqs"

# subprocess.run(["python", "hardware_specs.py"], check=True)

# os.chdir(juliabenchmarks)
# filenames = os.listdir(".")
# for name in filenames:
#     if "benchmarkutils" in name or not name.endswith(".jl"):
#         continue
#     subprocess.run(["julia", name], check=True)
# os.chdir("..")

# TODO: some issue with my laptop 
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
# 
# os.chdir(qutipbenchmarks)
# filenames = os.listdir(".")
# for name in filenames:
#     if "benchmarkutils" in name or not name.endswith(".py"):
#         continue
#     subprocess.run(["python", name], check=True)
# os.chdir("..")

os.chdir(dynamiqsbenchmarks)
filenames = os.listdir(".")
for name in filenames:
    if "benchmarkutils" in name or not name.endswith(".py"):
        continue
    subprocess.run(["python", name], check=True)
os.chdir("..")

# os.chdir(matlabbenchmarks)
# subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", 'run runall.m; quit;'], check=True)
# os.chdir("..")
