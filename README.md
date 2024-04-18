# Benchmarks for popular quantum-optics frameworks

> [!NOTE]  
> This is a fork of the repo linked below, which carefully implemented the benchmarks in QuTiP and QuantumOptics.jl
> I simply copied and reworked some of the code to work with dynamiqs.
> https://github.com/qojulia/QuantumOptics.jl-benchmarks


This repository collects a set of examples which can be used to compare different numerical quantum optics frameworks. The aim is to create implementations for the following frameworks:
* Dynamiqs (Python)
* QuTiP (Python)
* QuantumOptics.jl (Julia)


## Performing the benchmarks

### Setting up the testing environment

To reduce noise use dedicated cpu core for the benchmarks. I use a dedicated cluster node, and run `make.sh` using SLURM.

For julia, one still has to set the users home directory:

    HOME=/home/[username]

### Running the benchmarks

Now one can either manually run benchmarks of interest by simply executing the benchmark file or alternatively use the `run_all.py` script to run them all automatically. 
The bash script `make.sh` runs `run_all.py`, and also collects and plots the data. `make.sh` is compatible with SLURM. 

## More information to this repository

### The benchmark files

Every single benchmark file has more or less the same structure. A `setup` function that is used to create objects that is needed to perform the actual benchmark. The execution time of this code is not measured. The benchmark function which is called repeatedly with the previously calculated objects. And code needed to configure and perform the benchmark and check that the output is correct. Depending on each example, the time is measured that it takes to evaluate a function a certain number of times. This process is repeated a few times and the minimal time is used as benchmark time.

### Structure of the repository

* `benchmarks-{framework}`: where the benchmark files are.
* `results`: The results of the benchmarking is stored here as json files.
* `checks`: The output generated from QuantumOptics.jl is stored here and later on compared to QuTiP's output to make sure that the examples calculate the same things. At the moment the toolbox results are not checked.
* `results-collected`: All benchmarking results for each example are stored together in one json file.

### Helper scripts

* `run_all.py`: Benchmarks can be performed manually by executing the corresponding benchmark file or run automatically with the help of this script.
* `collect_results.py`: Collects the benchmarking results from the `results` directory, collects all benchmarks belonging to the same example into one file and writes the output as json into the `results-collected` directory.
* `extract_code.py`: Extracts the important benchmark code from all implementations and stores them in the `sourcecode` folder. Each file is executed to make sure that it really works.
* `hardware_specs.py`: Collects information about hardware and used software and stores it in the results directory.
* `plot_results-py`: Uses matplotlib to visualize the benchmark results stored in `results-collected`.
* `make.sh`: Runs ``run_all.py`, `collect_results.py`, `extract_code.py`, 
> [!NOTE]
> TODO: `make.sh` should also run `hardware_specs.py` and copy all files of interest into the correct website directories.
