#!/bin/bash
#SBATCH --job-name=thesis_purs_exp2_ml
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=CPUQ
#SBATCH --output=logs/purs_exp2_ml_%j.out
#SBATCH --error=logs/purs_exp2_ml_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no
# NOTE: run_purs_generate_ml.slurm must complete first.
# Submit with: sbatch --dependency=afterok:<generate_ml_jobid> run_purs_exp2_ml.slurm

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results_purs_ml

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python experiment2_idun.py \
    --recs_pkl idun_results_purs_ml/recommendations_fair_complete.pkl \
    --data_dir ../movielens/data \
    --out_dir  idun_results_purs_ml \
    --workers  $SLURM_CPUS_PER_TASK

echo "Job finished: $(date)"
