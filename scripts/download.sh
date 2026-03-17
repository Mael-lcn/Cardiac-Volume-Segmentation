#!/bin/bash
#SBATCH --job-name=dl_synapse
#SBATCH --partition=prepost           # Noeud avec accès Internet ouvert
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=../dump/dl_dataset.out
#SBATCH --error=../dump/dl_dataset.err
#SBATCH --account=iql@cpu

echo "=== Début du téléchargement ==="
date

module purge
# 1. On charge l'environnement
module load python/3.11.5
eval "$(conda shell.bash hook)"
conda activate ig3d

# 2. On se place dans le bon dossier
mkdir -p $SCRATCH/IG3D_CMRxRecon
cd $SCRATCH/IG3D_CMRxRecon

export SYNAPSE_AUTH_TOKEN="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3MzQzNzI2MSwiaWF0IjoxNzczNDM3MjYxLCJqdGkiOiIzMzY5MiIsInN1YiI6IjM1NzkwNDQifQ.HS_ibepN99udsfXBuUnASsIqRcAqvzc1bRJxaZWywjXqmGdrv2qnFslI_guSv1KiCXvxQqLRhMucjLBdvSH-DoD5ezpoJ60hF-_08U_BT8tGULAq-oo9u3OlxwLMNYE0YeMP-FI0nAV1Y4nEU_80mLv5r6e_wkJWYW6Ad5m3iXzefuWbQEtbDwF15QYNi1hjd2C27MFLsocJK_XiU7Q15SH4QV4U-ntirBxppqbM5LaFqmKawH0xswZwO5h8jGOz70k94gBMDdBXQ7r7TU_bZOWpQJ2zFJJSYuACHRM36XKYq8NoZA8hS0-4EGOjh1L8yeRUziI70UH9kIHK5VJYpA"

synapse get -q "SELECT * FROM syn63935430.1" --downloadLocation $SCRATCH/IG3D_CMRxRecon

echo "=== Téléchargement terminé ==="
date

