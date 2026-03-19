#!/bin/bash
#SBATCH --job-name=data_task
#SBATCH --qos=qos_cpu-t3
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --output=../dump/data_task.out
#SBATCH --error=../dump/data_task.err
#SBATCH --account=iql@cpu

module purge
module load python/3.11.5
conda activate ig3d

srun python $WORK/Cardiac-Volume-Segmentation/dataset/data_task.py -w 4
