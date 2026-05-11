#!/bin/bash
#SBATCH --job-name=thesis_exp1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:30:00              # 90 min — conservative for 5000 users
#SBATCH --partition=GPUQ             # or 'normal' — check with sinfo
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@stud.ntnu.no

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

mkdir -p logs idun_results

module purge
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/thesis_venv/bin/activate

cd $SLURM_SUBMIT_DIR

python experiment1_idun.py \
    --n_users 5000 \
    --workers $SLURM_CPUS_PER_TASK \
    --data_dir ../AMBAR \
    --out_dir idun_results \
    --seed 42

echo "Job finished: $(date)"
