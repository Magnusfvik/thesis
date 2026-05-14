#!/bin/bash
#SBATCH --job-name=thesis_pso
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # single-threaded — MOPSO is sequential
#SBATCH --mem=16G
#SBATCH --time=00:15:00              # ~5 min expected
#SBATCH --partition=CPUQ
#SBATCH --output=logs/pso_moo_%j.out
#SBATCH --error=logs/pso_moo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no
# NOTE: run_generate.slurm must complete before submitting this job.

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python pso_moo_idun.py \
    --recs_pkl idun_results/recommendations_fair_complete.pkl \
    --out_dir  idun_results

echo "Job finished: $(date)"
