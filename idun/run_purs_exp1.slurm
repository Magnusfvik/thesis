#!/bin/bash
#SBATCH --job-name=thesis_purs_exp1
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=CPUQ
#SBATCH --output=logs/purs_exp1_%j.out
#SBATCH --error=logs/purs_exp1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no

# NOTE: This job is self-contained — no generate step needed first.
#       SVD is trained internally and item embeddings are built on the fly.

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p logs idun_results_purs

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python experiment1_purs_idun.py \
    --n_users    5000 \
    --workers    $SLURM_CPUS_PER_TASK \
    --data_dir   ../AMBAR \
    --out_dir    idun_results_purs \
    --n_clusters 3 \
    --seed       42

echo "Job finished: $(date)"
