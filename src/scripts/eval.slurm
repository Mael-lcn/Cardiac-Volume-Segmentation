#!/bin/bash
#SBATCH --job-name=eval_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100
#SBATCH --output=../dump/eval_diffcmr.out
#SBATCH --error=../dump/eval_diffcmr.err

module purge
module load pytorch-gpu/py3/2.1.1
conda activate ig3d

task="diff" # "diff" ou "flow"

# Chemin vers tes logs d'inférence
export OPENAI_LOGDIR="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/$task/eval"

# Lancement distribué
srun python $WORK/ig3d/Cardiac-Volume-Segmentation/src/DiffCMR/inference.py \
    --model_path "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/$task/train/model60000.pt" \
    --val_pair_file "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/ValidationSet/pairs.txt" \
    --result_dir "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/log/$task/eval" \
    --model_type "diffusion" \
    --diffusion_steps 1000 \
    --timestep_respacing 100 \
    --batch_size 16 \
    --vote_num 4 \
    --num_workers 2 \
    --use_fp16 True