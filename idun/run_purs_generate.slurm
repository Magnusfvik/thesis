#!/bin/bash
#SBATCH --job-name=thesis_purs
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=CPUQ
#SBATCH --output=logs/purs_generate_%j.out
#SBATCH --error=logs/purs_generate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p logs idun_results_purs

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python generate_purs_inspired.py \
    --data_dir ../AMBAR \
    --out_dir  idun_results_purs \
    --n_users  5000 \
    --workers  $SLURM_CPUS_PER_TASK \
    --seed     42 \
    --n_clusters 3 \
    --checkpoint_every 500

echo "Job finished: $(date)"
