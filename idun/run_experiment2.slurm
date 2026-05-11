#!/bin/bash
#SBATCH --job-name=thesis_exp2
#SBATCH --account=magnusfv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # ILD computation only — fewer cores needed
#SBATCH --mem=32G
#SBATCH --time=00:30:00              # 30 min — fast, no SVD predictions
#SBATCH --partition=CPUQ
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no
# NOTE: run_generate.slurm must complete before submitting this job.
# Submit with dependency: sbatch --dependency=afterok:<generate_jobid> run_experiment2.slurm

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python experiment2_idun.py \
    --recs_pkl  idun_results/recommendations_fair_complete.pkl \
    --data_dir  ../AMBAR \
    --out_dir   idun_results \
    --workers   $SLURM_CPUS_PER_TASK

echo "Job finished: $(date)"
