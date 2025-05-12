#!/bin/bash
#
#SBATCH --partition=gpu_min32gb  # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min32gb        # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=speech      # Job name
#SBATCH -o slurm_%x.%j.out       # File containing STDOUT output
#SBATCH -e slurm_%x.%j.err       # File containing STDERR output. If ommited, use STDO


# Commands / scripts to run (e.g., python3 train.py)
# (...)

python train.py train_exp3_noaugment.yaml