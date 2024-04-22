#!/bin/bash
#SBATCH --job-name=dynamiqs-benchmarking
#SBATCH --output=dynamiqs-benchmarking.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=harsh.babla@yale.edu

module load miniconda
conda activate qi_cuda

cd src

python run_all.py
python collect_results.py
python plot_results.py