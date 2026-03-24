#!/bin/bash
#SBATCH --job-name=data_task
#SBATCH --qos=qos_cpu-t3
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=3:00:00
#SBATCH --output=../dump/train.out
#SBATCH --error=../dump/train.err
#SBATCH --account=iql@v100

module purge
module load python/3.11.5
conda activate ig3d

srun python $WORK/ig3d/Cardiac-Volume-Segmentation/dataset/data_task.py -w 8
