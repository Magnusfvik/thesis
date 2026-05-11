#!/bin/bash
#SBATCH --job-name=thesis_gen
#SBATCH --account=magnusfv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # 32 workers — adjust up to 64 if available
#SBATCH --mem=64G                    # 64 GB RAM — safe for 5000 users
#SBATCH --time=02:00:00              # 2-hour wall time (50 min expected)
#SBATCH --partition=CPUQ             # CPU partition — use GPUQ only if you need GPU
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

mkdir -p logs idun_results

# ---------------------------------------------------------------------------
# Load Conda environment (recommended by IDUN tutorial)
# ---------------------------------------------------------------------------
module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

# If thesis_env doesn't exist yet, create it first (only needed once):
# module load Anaconda3/2023.09-0
# conda create -n thesis_env python=3.11 -y
# conda activate thesis_env
# pip install numpy pandas scikit-surprise scipy matplotlib seaborn

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

python generate_complete_idun.py \
    --n_users 5000 \
    --workers $SLURM_CPUS_PER_TASK \
    --data_dir ../AMBAR \
    --out_dir idun_results \
    --seed 42 \
    --checkpoint_every 500

echo "Job finished: $(date)"
