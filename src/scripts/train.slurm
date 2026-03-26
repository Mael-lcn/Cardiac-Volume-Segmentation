#!/bin/bash
#SBATCH --job-name=train_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100
#SBATCH --output=../dump/train.out
#SBATCH --error=../dump/train.err

module purge
module load pytorch-gpu/py3/2.1.1
conda activate ig3d

# Lancement distribué
srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/train_task1_diff.py \
    --trainpairfile "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt" \
    --logdir "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/logs_run1" \
    --use_fp16 True \
    --num_workers 2 \
    --batch_size 8
