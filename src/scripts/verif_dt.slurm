#!/bin/bash
#SBATCH --job-name=verif_dt
#SBATCH --qos=qos_cpu-t3
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=../dump/verif_dt.out
#SBATCH --error=../dump/verif_dt.err
#SBATCH --account=iql@cpu

module purge
module load python/3.11.5
conda activate ig3d

srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/CMRxRecon.py
