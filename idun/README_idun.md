# IDUN Setup Guide

## Files in this folder

| Script | What it does | Depends on |
|---|---|---|
| `generate_complete_idun.py` | Generates top-10 recs for all users × 11 alpha values | nothing |
| `experiment1_idun.py` | Unimodal hypothesis test (distance vs rating) | nothing |
| `experiment2_idun.py` | Multi-objective serendipity analysis across alpha values | `generate_complete_idun.py` |
| `run_generate.slurm` | SLURM job for generate_complete_idun.py | — |
| `run_experiment1.slurm` | SLURM job for experiment1_idun.py | — |
| `run_experiment2.slurm` | SLURM job for experiment2_idun.py | — |

## Correct run order

```
generate_complete_idun.py  ──┐
                             ├──► experiment2_idun.py
experiment1_idun.py  ────────┘ (independent, can run at same time as generate)
```

Steps 1 and 2 can be submitted simultaneously. Step 3 must wait for step 1.

---

## One-time setup on IDUN

```bash
# SSH into IDUN
ssh username@idun.hpc.ntnu.no

# Create Python environment (only needed once)
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv $HOME/thesis_venv
source $HOME/thesis_venv/bin/activate
pip install numpy pandas scikit-surprise scipy matplotlib seaborn
```

## Transfer files to IDUN

Run from your **local machine**:

```bash
# Transfer AMBAR dataset (large — do once, takes a few minutes)
rsync -avz --progress \
    /Users/Magnusvik/dev/thesis/thesis/AMBAR/ \
    username@idun.hpc.ntnu.no:thesis/thesis/AMBAR/

# Transfer IDUN scripts
rsync -avz --progress \
    /Users/Magnusvik/dev/thesis/thesis/idun/ \
    username@idun.hpc.ntnu.no:thesis/thesis/idun/
```

## Edit the SLURM scripts

In **all three** `.slurm` files, replace:
- `YOUR_ACCOUNT` → your NTNU HPC project account (run `id` on IDUN to find it, or ask supervisor)
- `YOUR_EMAIL@stud.ntnu.no` → your email
- `--partition=GPUQ` → check available partitions with `sinfo` on IDUN

## Submit jobs

```bash
# SSH into IDUN
ssh username@idun.hpc.ntnu.no
cd thesis/thesis/idun
mkdir -p logs

# Submit step 1 (recommendation generation) and step 2 (experiment 1) in parallel
GEN_JOB=$(sbatch --parsable run_generate.slurm)
sbatch run_experiment1.slurm
echo "Generate job ID: $GEN_JOB"

# Submit experiment 2 — runs automatically after generate completes
sbatch --dependency=afterok:$GEN_JOB run_experiment2.slurm
```

The `--dependency=afterok` ensures experiment2 only starts once `generate_complete_idun.py` has finished successfully.

## Monitor progress

```bash
squeue -u $USER                        # show your jobs
tail -f logs/generate_<jobid>.out      # live log for generate job
tail -f logs/exp1_<jobid>.out          # live log for experiment 1
```

## Retrieve results

Run from your **local machine** after all jobs complete:

```bash
rsync -avz --progress \
    username@idun.hpc.ntnu.no:thesis/thesis/idun/idun_results/ \
    /Users/Magnusvik/dev/thesis/thesis/idun/idun_results/
```

## Expected runtimes (32 CPUs, 5000 users)

| Job | Expected time |
|---|---|
| `run_generate.slurm` | ~50 min |
| `run_experiment1.slurm` | ~30 min |
| `run_experiment2.slurm` | ~5 min (after generate finishes) |

## Output files

All results land in `idun_results/`:

| File | Produced by |
|---|---|
| `recommendations_fair_complete.pkl` | generate_complete_idun |
| `generation_info.pkl` | generate_complete_idun |
| `exp1_detailed_results.csv` | experiment1_idun |
| `exp1_bin_statistics.csv` | experiment1_idun |
| `exp1_regression.csv` | experiment1_idun |
| `exp1_summary.txt` | experiment1_idun |
| `exp1_unimodal_relationship.png` | experiment1_idun |
| `exp2_results.csv` | experiment2_idun |
| `exp2_summary.txt` | experiment2_idun |
| `exp2_serendipity_curve.png` | experiment2_idun |
| `exp2_tradeoff_frontier.png` | experiment2_idun |

## Scaling options

To run on more users, edit the `--n_users` argument in the SLURM scripts:

| Users | RAM needed | Expected time (32 cores) |
|---|---|---|
| 5,000 | 32 GB | ~50 min |
| 10,000 | 48 GB | ~100 min |
| 30,816 (all) | 64 GB | ~5 hours |

## Useful IDUN commands

```bash
sinfo                        # show available partitions and node types
squeue -u $USER              # your running/pending jobs
scancel <jobid>              # cancel a job
sacct -j <jobid>             # accounting info after job completes
```
