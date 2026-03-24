import os
import time
import argparse
import multiprocessing
import traceback
import h5py
import gc
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from loadFun import load_h5_slice, multicoilkdata2img_slice



def paddingZero_np(np_data: np.ndarray, target_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Applique un padding (remplissage de zéros) ou un center-cropping dynamique sur un tenseur.
    Alloue directement la matrice cible et injecte les données, évitant la duplication en RAM.
    """
    shape = list(np_data.shape)
    H, W = shape[-2], shape[-1]

    if H == target_shape[0] and W == target_shape[1]:
        return np_data

    shape[-2], shape[-1] = target_shape[0], target_shape[1]
    padded_data = np.zeros(shape, dtype=np_data.dtype)

    src_h_start = max((H - target_shape[0]) // 2, 0)
    src_w_start = max((W - target_shape[1]) // 2, 0)
    src_h_end = src_h_start + min(H, target_shape[0])
    src_w_end = src_w_start + min(W, target_shape[1])

    dst_h_start = max((target_shape[0] - H) // 2, 0)
    dst_w_start = max((target_shape[1] - W) // 2, 0)
    dst_h_end = dst_h_start + min(H, target_shape[0])
    dst_w_end = dst_w_start + min(W, target_shape[1])

    padded_data[..., dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
        np_data[..., src_h_start:src_h_end, src_w_start:src_w_end]

    return padded_data


def process_single_slice(args: Tuple) -> Dict[str, Any]:
    """
    Pipeline de traitement pour UNE unique tranche (Slice) d'IRM.
    Gère la RAM de façon chirurgicale avec des purges (del) in-place.
    """
    full_path, mask_path, slice_idx, axis_name, item, save_full_dir, save_04_dir, coilInfo = args
    result = {"status": "SUCCESS", "item": item, "slice_idx": slice_idx, "axis": axis_name, "msg": ""}

    try:
        # 1. LAZY LOADING
        raw_full = load_h5_slice(full_path, slice_idx, dataset_name='kspace_full')
        raw_mask = load_h5_slice(mask_path, slice_idx)

        # 2. CONVERSION COMPLEXE IN-PLACE
        data_full = np.empty(raw_full.shape, dtype=np.complex64)
        if raw_full.dtype.names is not None and 'real' in raw_full.dtype.names:
            data_full.real, data_full.imag = raw_full['real'], raw_full['imag']
        else:
            data_full[:] = raw_full
        del raw_full 
        
        mask_04 = np.empty(raw_mask.shape, dtype=np.complex64)
        if raw_mask.dtype.names is not None and 'real' in raw_mask.dtype.names:
            mask_04.real, mask_04.imag = raw_mask['real'], raw_mask['imag']
        else:
            mask_04[:] = raw_mask
        del raw_mask

        # 3. SOUS-ÉCHANTILLONNAGE & PADDING
        data_04 = data_full * mask_04
        del mask_04
        
        data_full_padded = paddingZero_np(data_full, (512, 512))
        del data_full
        data_04_padded = paddingZero_np(data_04, (512, 512))
        del data_04

        # 4. IFFT (Transformation Image)
        imgs_full = multicoilkdata2img_slice(data_full_padded)
        del data_full_padded
        imgs_04 = multicoilkdata2img_slice(data_04_padded)
        del data_04_padded

        # 5. NORMALISATION (In-place strict)
        max_f = np.amax(imgs_full, axis=(1, 2), keepdims=True)
        np.divide(imgs_full, max_f, out=imgs_full, where=max_f!=0)
        del max_f

        max_4 = np.amax(imgs_04, axis=(1, 2), keepdims=True)
        np.divide(imgs_04, max_4, out=imgs_04, where=max_4!=0)
        del max_4

        # 6. SAUVEGARDE CHUNKÉE DANS LE SOUS-DOSSIER PATIENT
        file_name = f"{item}_{coilInfo}_{axis_name}_slice{slice_idx:02d}.npy"
        np.save(os.path.join(save_full_dir, file_name), imgs_full)
        np.save(os.path.join(save_04_dir, file_name), imgs_04)
        
        del imgs_full, imgs_04
        gc.collect()
        
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["msg"] = f"[ERREUR FATALE] {item} ({axis_name} tranche {slice_idx}):\n{traceback.format_exc()}"
        return result


def generate_slice_tasks(patient_dir: str, item: str, save_dir: str, coilInfo: str) -> List[Tuple]:
    """
    Inspecte un patient et génère les tâches.
    Inclus la logique de SAUT (Skip) si le patient a déjà été traité.
    """
    tasks = []
    
    # Création des chemins incluant le nom du patient (ex: output/FullSample/P001)
    save_full_dir = os.path.join(save_dir, "FullSample", item)
    save_04_dir = os.path.join(save_dir, "AccFactor04", item)

    # --- LOGIQUE ANTI-DOUBLON (SKIP LOGIC) ---
    if os.path.exists(save_full_dir) and os.path.exists(save_04_dir):
        # On vérifie si les dossiers contiennent déjà des fichiers .npy
        full_files = [f for f in os.listdir(save_full_dir) if f.endswith('.npy')]
        acc_files = [f for f in os.listdir(save_04_dir) if f.endswith('.npy')]
        
        if len(full_files) > 0 and len(acc_files) > 0:
            print(f"  [SKIP] {item} ignoré : Déjà traité ({len(full_files)} fichiers existants).")
            return [] # On retourne une liste vide, le patient ne sera pas recalculé

    # Si le dossier n'existe pas ou est vide, on le crée
    os.makedirs(save_full_dir, exist_ok=True)
    os.makedirs(save_04_dir, exist_ok=True)

    mask_dir = patient_dir.replace('FullSample', 'Mask_Task1')
    axes = [
        ("lax", os.path.join(patient_dir, "cine_lax.mat"), os.path.join(mask_dir, "cine_lax_mask_Uniform4.mat")),
        ("sax", os.path.join(patient_dir, "cine_sax.mat"), os.path.join(mask_dir, "cine_sax_mask_Uniform4.mat"))
    ]

    for axis_name, full_path, mask_path in axes:
        if os.path.exists(full_path) and os.path.exists(mask_path):
            try:
                with h5py.File(full_path, 'r') as f:
                    num_slices = f['kspace_full'].shape[1] 
                
                for slice_idx in range(num_slices):
                    tasks.append((full_path, mask_path, slice_idx, axis_name, item, save_full_dir, save_04_dir, coilInfo))
            except Exception as e:
                print(f"  [ATTENTION] {item} : Impossible d'inspecter l'axe {axis_name} -> {e}")
                
    if len(tasks) > 0:
        print(f"  [AJOUT] {item} planifié : {len(tasks)} tranches à traiter.")
        
    return tasks


def generate_training_pairs(base_path: str, acc_factor: str = "AccFactor04") -> Tuple[int, str]:
    """ 
    Génère le fichier texte de mapping en parcourant récursivement les sous-dossiers patients. 
    """
    imgs_dir_base = os.path.join(base_path, acc_factor)
    full_sample_dir_base = os.path.join(base_path, 'FullSample')
    pairs_count = 0

    if not os.path.exists(imgs_dir_base) or not os.path.exists(full_sample_dir_base):
        return 0, "[ERREUR] Dossiers cibles introuvables."

    file_path = os.path.join(base_path, f"{acc_factor}_training_pair.txt")

    try:
        with open(file_path, "w") as file_obj:
            # Parcours de chaque dossier patient (P001, P002...)
            for patient_folder in sorted(os.listdir(imgs_dir_base)):
                patient_imgs_dir = os.path.join(imgs_dir_base, patient_folder)
                patient_gt_dir = os.path.join(full_sample_dir_base, patient_folder)

                if not os.path.isdir(patient_imgs_dir):
                    continue # Ignore les fichiers parasites à la racine

                # Parcours des fichiers .npy dans le dossier du patient
                for img_name in sorted(os.listdir(patient_imgs_dir)):
                    if img_name.endswith('.npy'):
                        img_path = os.path.join(patient_imgs_dir, img_name)
                        gt_path = os.path.join(patient_gt_dir, img_name)
                        
                        if os.path.exists(gt_path):
                            file_obj.write(f"{img_path} {gt_path}\n")
                            pairs_count += 1
                            
        return pairs_count, file_path
    except Exception as e:
        return 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Prétraitement HPC Lazy-Loading et Chunking")
    parser.add_argument('-i', "--input", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data_pre/home2/Raw_data/MICCAIChallenge2024/ChallengeData")
    parser.add_argument('-o', "--output", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data")
    parser.add_argument('-w', "--workers", type=int, default=max(1, multiprocessing.cpu_count()-1))

    args = parser.parse_args()
    coilInfo = 'MultiCoil'
    dir_path = os.path.join(args.input, coilInfo, 'Cine', 'TrainingSet', 'FullSample')

    if not os.path.isdir(dir_path):
        print(f"[ERREUR CRITIQUE] Dossier source introuvable : {dir_path}")
        return

    print("=========================================================")
    print(" 1. SCAN DES PATIENTS ET PLANIFICATION (AVEC SKIP-LOGIC) ")
    print("=========================================================")
    all_slice_tasks = []

    for item in sorted(os.listdir(dir_path)):
        patient_dir = os.path.join(dir_path, item)
        if os.path.isdir(patient_dir):
            all_slice_tasks.extend(generate_slice_tasks(patient_dir, item, args.output, coilInfo))

    total_tasks = len(all_slice_tasks)
    if total_tasks == 0:
        print("\n[INFO] Toutes les données ont déjà été traitées (ou dossier vide). Fin du script.")
        return

    print(f"\n-> TOTAL : {total_tasks} nouvelles tranches à traiter réparties sur {args.workers} cœurs.")
    
    start_time = time.time()
    success_count = 0
    errors = 0

    print("\n=========================================================")
    print(" 2. EXECUTION MULTIPROCESSING (RECONSTRUCTION 2D)        ")
    print("=========================================================")
    with multiprocessing.Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_slice, all_slice_tasks), total=total_tasks, desc="Progression globale"):
            if result["status"] == "SUCCESS":
                success_count += 1
            else:
                errors += 1
                # Utilisation de tqdm.write pour ne pas casser la barre de progression
                tqdm.write(result["msg"])

    print("\n=========================================================")
    print(" 3. CONSOLIDATION DU MAPPING PYTORCH                     ")
    print("=========================================================")
    pairs_count, pair_file_path = generate_training_pairs(args.output)
    print(f"  -> Fichier de paires mis à jour : {pair_file_path}")

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"""
    =========================================================
                    RAPPORT D'EXECUTION FINALE              
    =========================================================
    Temps de calcul total   : {int(hours)}h {int(minutes)}m {int(seconds)}s
    Tranches calculées      : {success_count} / {total_tasks}
    Paires d'entraînement   : {pairs_count} (Total dataset)
    Erreurs fatales         : {errors}
    =========================================================
        """)

if __name__ == "__main__":
    main()
