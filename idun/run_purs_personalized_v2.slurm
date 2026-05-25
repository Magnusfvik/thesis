#!/bin/bash
#SBATCH --job-name=thesis_pso_per_v2
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # MOPSO is sequential
#SBATCH --mem=32G
#SBATCH --time=02:00:00              # v2: 100 particles × 500 iter → ~60-90 min
#SBATCH --partition=CPUQ
#SBATCH --output=logs/pso_per_v2_%j.out
#SBATCH --error=logs/pso_per_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=magnusfv@stud.ntnu.no
# Prerequisite: run_purs_generate.slurm must have completed (produces
# idun_results_purs/recommendations_fair_complete.pkl and
# idun_results_purs/pso_moo_front.csv)

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs idun_results_purs

module purge
module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate thesis_env

cd $SLURM_SUBMIT_DIR

python pso_moo_personalized_v2_idun.py \
    --recs_pkl     idun_results_purs/recommendations_fair_complete.pkl \
    --single_front idun_results_purs/pso_moo_front.csv \
    --out_dir      idun_results_purs

echo "Job finished: $(date)"
