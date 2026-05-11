#!/bin/bash
#SBATCH --job-name=thesis_gen
#SBATCH --account=YOUR_ACCOUNT       # replace with your NTNU project account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # 32 workers — adjust up to 64 if available
#SBATCH --mem=64G                    # 64 GB RAM — safe for 5000 users
#SBATCH --time=02:00:00              # 2-hour wall time (50 min expected)
#SBATCH --partition=GPUQ             # or 'hugemem' / 'normal' — check idun docs
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@stud.ntnu.no

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

mkdir -p logs idun_results

# ---------------------------------------------------------------------------
# Load Python environment
# ---------------------------------------------------------------------------
module purge
module load Python/3.11.3-GCCcore-12.3.0   # adjust version to what IDUN has

# Activate your virtual environment (copy it to IDUN or use pip install below)
source $HOME/thesis_venv/bin/activate

# If venv doesn't exist yet, create it first (only needed once):
# python3 -m venv $HOME/thesis_venv
# source $HOME/thesis_venv/bin/activate
# pip install numpy pandas scikit-surprise scipy matplotlib

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
