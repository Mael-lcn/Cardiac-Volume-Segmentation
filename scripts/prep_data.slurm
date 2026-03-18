#!/bin/bash
#SBATCH --job-name=prep_cmrx
#SBATCH --partition=cpu_p1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --output=../dump/prep_data.out
#SBATCH --error=../dump/prep_data.err
#SBATCH --account=iql@cpu

set -e # Stop si une commande échoue (ex: disque plein)

echo "=== Début du traitement (Direct Mode) ==="
date

cd $SCRATCH/IG3D_CMRxRecon

# 1. Fusion des fragments vers le ZIP final
echo "[1/5] Fusion des 92 fragments vers ChallengeData.zip..."
cat ChallengeData.zip-part-* > ChallengeData.zip

# 2. Vérification MD5 
echo "[2/5] Vérification MD5 du ZIP final..."
md5sum ChallengeData.zip
# Rappel post : e45e42e4aeb6ef1870fedcf1454ec262 et normal: 31adb082971e2aa8149da9df045b43f5

# 3. Décompression
echo "[3/5] Décompression vers extracted_data/..."
mkdir -p extracted_data
unzip -q ChallengeData.zip -d extracted_data/

# 4. Nettoyage des fichiers corrompus
echo "[4/5] Suppression des données anormales..."
BASE_GT="extracted_data/CMRxRecon2024_Val_TestDataset/GroundTruth"
rm -f "$BASE_GT/Cine/TrainingSet/FullSample/P100/cinesax.mat" \
      "$BASE_GT/Cine/TrainingSet/FullSample/P109/cinesax.mat" \
      "$BASE_GT/Aorta/TrainingSet/FullSample/P065/aortasag.mat" \
      "$BASE_GT/Aorta/TrainingSet/FullSample/P077/aortasag.mat" \
      "$BASE_GT/Aorta/TrainingSet/FullSample/P105/aortatra.mat" \
      "$BASE_GT/Aorta/TrainingSet/FullSample/P117/aortatra.mat" \
      "$BASE_GT/Tagging/TrainingSet/FullSample/P087/tagging.mat" \
      "$BASE_GT/Tagging/TrainingSet/FullSample/P105/tagging.mat"

# 5. Nettoyage final (Une fois que tout est OK)
echo "[5/5] Ménage : Suppression du ZIP et des fragments..."
rm ChallengeData.zip-part-*
rm ChallengeData.zip

echo "=== Traitement terminé avec succès ! ==="
date
