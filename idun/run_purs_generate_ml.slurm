#!/bin/bash
#SBATCH --job-name=thesis_purs_ml
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=CPUQ
#SBATCH --output=logs/purs_ml_generate_%j.out
#SBATCH --error=logs/purs_ml_generate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results_purs_ml

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python generate_purs_inspired.py \
    --data_dir ../movielens/data \
    --out_dir  idun_results_purs_ml \
    --n_users  6040 \
    --workers  $SLURM_CPUS_PER_TASK \
    --seed     42 \
    --n_clusters 3 \
    --checkpoint_every 500

echo "Job finished: $(date)"
