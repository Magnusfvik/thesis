#!/bin/bash
#SBATCH --job-name=thesis_pso_per
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # single-threaded — MOPSO is sequential
#SBATCH --mem=32G                    # 5000-user interpolators need more RAM
#SBATCH --time=00:30:00              # ~10-15 min expected for 5000D
#SBATCH --partition=CPUQ
#SBATCH --output=logs/pso_per_%j.out
#SBATCH --error=logs/pso_per_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no
# NOTE: run_pso_moo.slurm must complete before submitting this job
# (needs idun_results/pso_moo_front.csv for the comparison plot).

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python pso_moo_personalized_idun.py \
    --recs_pkl     idun_results/recommendations_fair_complete.pkl \
    --single_front idun_results/pso_moo_front.csv \
    --out_dir      idun_results

echo "Job finished: $(date)"
