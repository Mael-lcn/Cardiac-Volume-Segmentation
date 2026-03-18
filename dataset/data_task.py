import os
import time
import argparse
import multiprocessing
import traceback
import numpy as np
from tqdm import tqdm
from loadFun import loadmat, multicoilkdata2img 



def paddingZero_np(np_data, target_shape):
    """
    Applique un padding avec des zéros pour forcer les dimensions de l'image.
    """
    shape = np_data.shape
    H, W = shape[-2], shape[-1]
    padding_H = target_shape[0] - H
    padding_W = target_shape[1] - W
    
    if len(shape) == 4:
        padding_size = ((0, 0), (0, 0), (padding_H // 2, padding_H - padding_H // 2), (padding_W // 2, padding_W - padding_W // 2))
    else:
        padding_size = ((0, 0), (0, 0), (0, 0), (padding_H // 2, padding_H - padding_H // 2), (padding_W // 2, padding_W - padding_W // 2))
    
    return np.pad(np_data, padding_size, mode='constant')


def process_single_patient(args):
    """
    Fonction de traitement (worker) adaptative pour le format 2024.
    Applique le masque directement sur l'Espace K complet.
    """
    patient_dir, item, output, coilInfo = args
    result = {
        "status": "SUCCESS", 
        "patient": item,
        "files_saved": 0,
        "msg": ""
    }

    try:
        # Chemins des données brutes (FullSample)
        lax_full_path = os.path.join(patient_dir, "cine_lax.mat")
        sax_full_path = os.path.join(patient_dir, "cine_sax.mat")

        # Déduction des chemins des masques (Mask_Task1)
        mask_dir = patient_dir.replace('FullSample', 'Mask_Task1')
        lax_mask_path = os.path.join(mask_dir, "cine_lax_mask_Uniform4.mat")
        sax_mask_path = os.path.join(mask_dir, "cine_sax_mask_Uniform4.mat")

        # Création des dossiers cibles
        save_full_dir = os.path.join(output, "FullSample")
        save_04_dir = os.path.join(output, "AccFactor04")
        os.makedirs(save_full_dir, exist_ok=True)
        os.makedirs(save_04_dir, exist_ok=True)

        def process_axis(full_path, mask_path, axis_name):
            if not (os.path.exists(full_path) and os.path.exists(mask_path)):
                return 0

            # 1. Chargement des données et du masque
            data_full = loadmat(full_path)["kspace_full"]
            masks = loadmat(mask_path)

            # Extraction dynamique de la matrice du masque
            mask_key = next((k for k in masks.keys() if not k.startswith('__')), None)
            if mask_key is None:
                raise KeyError(f"Matrice de masque introuvable dans {mask_path}")
            mask_04 = masks[mask_key]

            # 2. Application du sous-échantillonnage
            data_04 = data_full * mask_04

            # 3. Application du Padding (512x512)
            data_full_padded = paddingZero_np(data_full, (512, 512))
            data_04_padded = paddingZero_np(data_04, (512, 512))

            # 4. Reconstruction dans le domaine spatial
            imgs_full = multicoilkdata2img(data_full_padded)
            imgs_04 = multicoilkdata2img(data_04_padded)

            # 5. Normalisation vectorisée (Division par le maximum local)
            max_f = np.amax(imgs_full, axis=(2, 3), keepdims=True)
            imgs_full = np.divide(imgs_full, max_f, out=np.zeros_like(imgs_full), where=max_f!=0)

            max_4 = np.amax(imgs_04, axis=(2, 3), keepdims=True)
            imgs_04 = np.divide(imgs_04, max_4, out=np.zeros_like(imgs_04), where=max_4!=0)

            # 6. Extraction et sauvegarde des coupes 2D au format Numpy
            files_count = 0
            frame_num, slice_num, H, W = imgs_full.shape
            for i in range(frame_num):
                for j in range(slice_num):
                    np.save(os.path.join(save_full_dir, f"{item}_{coilInfo}_{axis_name}_{i}_{j}_{H}_{W}.npy"), imgs_full[i, j])
                    np.save(os.path.join(save_04_dir, f"{item}_{coilInfo}_{axis_name}_{i}_{j}_{H}_{W}.npy"), imgs_04[i, j])
                    files_count += 2
                    
            return files_count

        # Exécution du traitement pour les axes longs (LAX) et courts (SAX)
        saved_lax = process_axis(lax_full_path, lax_mask_path, "lax")
        saved_sax = process_axis(sax_full_path, sax_mask_path, "sax")
        
        result["files_saved"] = saved_lax + saved_sax

        if result["files_saved"] == 0:
            result["status"] = "WARNING"
            result["msg"] = f"[ATTENTION] Patient {item} : Fichiers originaux ou masques introuvables."
        else:
            result["msg"] = f"[SUCCES] Patient {item} traité ({result['files_saved']} fichiers générés)."
            
        return result

    except Exception as e:
        error_details = traceback.format_exc()
        result["status"] = "ERROR"
        result["msg"] = f"\n[ERREUR FATALE] Interruption sur le patient {item}.\nTrace :\n{error_details}"
        return result


def generate_training_pairs(base_path, acc_factor="AccFactor04"):
    """
    Génère le fichier texte associant les données d'entrée aux vérités terrains.
    """
    imgs_dir = os.path.join(base_path, acc_factor)
    full_sample_dir = os.path.join(base_path, 'FullSample')
    pairs_count = 0

    if not os.path.exists(imgs_dir) or not os.path.exists(full_sample_dir):
        return 0, "[ERREUR] Dossiers cibles introuvables pour la génération des paires."

    file_path = os.path.join(base_path, f"{acc_factor}_rMax_512_training_pair.txt")

    try:
        with open(file_path, "w") as file_obj:
            for img_name in os.listdir(imgs_dir):
                if img_name.endswith('.npy'):
                    img_path = os.path.join(imgs_dir, img_name)
                    GT_path = os.path.join(full_sample_dir, img_name)
                    
                    if os.path.exists(GT_path):
                        file_obj.write(f"{img_path} {GT_path}\n")
                        pairs_count += 1
                        
        return pairs_count, file_path
    except Exception as e:
        return 0, f"[ERREUR] Echec lors de la création du fichier de paires: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description="Prétraitement HPC")
    parser.add_argument('-i', "--input", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/IG3D_CMRxRecon/data_pre/home2/Raw_data/MICCAIChallenge2024/ChallengeData", help="Dossier racine contenant les données brutes")
    parser.add_argument('-o', "--output", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/IG3D_CMRxRecon/data", help="Dossier de destination pour les fichiers .npy")
    parser.add_argument('-w', "--workers", type=int, default=max(1, multiprocessing.cpu_count()-1), 
                        help="Nombre de processus à utiliser (Défaut : coeurs disponibles - 1)")

    args = parser.parse_args()

    modalityName = 'Cine'
    coilInfo = 'MultiCoil'

    # Cible uniquement le dossier FullSample pour identifier les patients
    tasks = []
    dir_path = os.path.join(args.input, coilInfo, modalityName, 'TrainingSet', 'FullSample')

    if not os.path.isdir(dir_path):
        print(f"[ERREUR CRITIQUE] Le dossier FullSample est introuvable à l'emplacement : {dir_path}")
        return

    for item in os.listdir(dir_path)[:4]:
        patient_dir = os.path.join(dir_path, item)
        if os.path.isdir(patient_dir):
            tasks.append((patient_dir, item, args.output, coilInfo))

    total_patients = len(tasks)
    if total_patients == 0:
        print("[INFO] Aucun patient détecté. Fin du programme.")
        return

    print(f"Demarrage du pool multiprocessing avec {args.workers} processus.")
    print(f"{total_patients} patients soumis au traitement.")

    start_time = time.time()
    stats = {
        "success": 0,
        "warnings": 0,
        "errors": 0,
        "total_files_saved": 0
    }

    # Lancement du traitement parallèle asynchrone
    with multiprocessing.Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_patient, tasks), total=total_patients, desc="Traitement IRM"):
            if result["status"] == "SUCCESS":
                stats["success"] += 1
                stats["total_files_saved"] += result["files_saved"]
            elif result["status"] == "WARNING":
                stats["warnings"] += 1
                tqdm.write(result["msg"])
            elif result["status"] == "ERROR":
                stats["errors"] += 1
                tqdm.write(result["msg"])

    print("\nGeneration du fichier de mapping (Paires) en cours...")
    pairs_count, pairs_msg = generate_training_pairs(args.output, acc_factor="AccFactor04")

    # Calcul des métriques de temps
    total_time = time.time() - start_time
    avg_time_per_patient = total_time / total_patients if total_patients > 0 else 0
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    summary = f"""
        =========================================================
                        RAPPORT D'EXECUTION FINALE              
        =========================================================
        Temps total écoulé      : {int(hours)}h {int(minutes)}m {int(seconds)}s
        Temps moyen / patient   : {avg_time_per_patient:.2f} secondes
        Threads (Workers)       : {args.workers}
        ---------------------------------------------------------
        Patients traités        : {total_patients}
        Succès                  : {stats["success"]}
        Avertissements          : {stats["warnings"]}
        Erreurs                 : {stats["errors"]}
        ---------------------------------------------------------
        Fichiers .npy générés   : {stats["total_files_saved"]}
        Paires d'entraînement   : {pairs_count}
        =========================================================
    """
    print(summary)


if __name__ == "__main__":
    main()
