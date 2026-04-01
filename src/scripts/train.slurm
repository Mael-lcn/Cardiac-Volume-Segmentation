#!/bin/bash
#SBATCH --job-name=train_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100

module purge
module load pytorch-gpu/py3/2.1.1
conda activate ig3d

task="diff" # ou "flow"
export OPENAI_LOGDIR="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/$task/logs_run1"
mkdir -p $OPENAI_LOGDIR

srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/train_task1_$task.py \
    --model_type "diffusion" \
    --trainpairfile "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt" \
    --valpairfile "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt" \
    --lr 1e-4 \
    --batch_size 8 \
    --microbatch -1 \
    --diffusion_steps 1000 \
    --log_interval 1000 \
    --save_interval 10000 \
    --use_fp16 True \
    --run_without_test False
