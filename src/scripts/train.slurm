#!/bin/bash
#SBATCH --job-name=train_diffcmr
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=24:00:00             
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                # Demande de l'intégralité des 8 V100
#SBATCH --ntasks-per-node=4         # 8 tâches MPI (1 processus gère 1 GPU)
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --account=iql@v100
#SBATCH --output=../dump/train.out
#SBATCH --error=../dump/train.err

module purge
module load pytorch-gpu/py3/2.1.1

# srun va automatiquement lancer 8 processus synchronisés (1 par GPU)
srun python train_task1_diff.py \
    --trainpairfile "/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data/TrainingSet/pairs.txt" \
    --use_fp16 True \
    --num_workers 3
