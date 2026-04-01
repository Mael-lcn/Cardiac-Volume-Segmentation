import os
import argparse
import multiprocessing
import traceback
import h5py
import gc
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from loadFun import load_h5_slice, multicoilkdata2img_slice



def padding_zero_512(np_data: np.ndarray, target_shape: Tuple[int, int] = (512, 512)) -> np.ndarray:
    shape = list(np_data.shape)
    h_orig, w_orig = shape[-2], shape[-1]

    if h_orig == target_shape[0] and w_orig == target_shape[1]:
        return np_data

    shape[-2], shape[-1] = target_shape[0], target_shape[1]
    padded_data = np.zeros(shape, dtype=np_data.dtype)

    sh, sw = min(h_orig, target_shape[0]), min(w_orig, target_shape[1])
    src_h = max((h_orig - target_shape[0]) // 2, 0)
    src_w = max((w_orig - target_shape[1]) // 2, 0)
    dst_h = max((target_shape[0] - h_orig) // 2, 0)
    dst_w = max((target_shape[1] - w_orig) // 2, 0)

    padded_data[..., dst_h:dst_h+sh, dst_w:dst_w+sw] = \
        np_data[..., src_h:src_h+sh, src_w:src_w+sw]

    return padded_data

def process_single_slice(args: Tuple) -> Dict[str, Any]:
    full_path, mask_path, slice_idx, axis_name, item, save_full_dir, save_04_dir, coil_info = args
    result = {"status": "SUCCESS", "item": item, "slice": slice_idx, "axis": axis_name, "num_frames": 0, "msg": ""}

    try:
        raw_full = load_h5_slice(full_path, slice_idx, dataset_name='kspace_full')
        
        try:
            raw_mask = load_h5_slice(mask_path, slice_idx)
        except Exception:
            with h5py.File(mask_path, 'r') as f_m:
                m_key = 'mask' if 'mask' in f_m else 'mask_indices'
                raw_mask = np.array(f_m[m_key])

        data_full = np.empty(raw_full.shape, dtype=np.complex64)
        if raw_full.dtype.names and 'real' in raw_full.dtype.names:
            data_full.real, data_full.imag = raw_full['real'], raw_full['imag']
        else:
            data_full[:] = raw_full
        
        mask_04 = np.empty(raw_mask.shape, dtype=np.complex64)
        if raw_mask.dtype.names and 'real' in raw_mask.dtype.names:
            mask_04.real, mask_04.imag = raw_mask['real'], raw_mask['imag']
        else:
            mask_04[:] = raw_mask

        data_04 = data_full * mask_04

        # 1. Transformation en image (La fonction corrigée est appelée ici)
        imgs_full = multicoilkdata2img_slice(padding_zero_512(data_full))
        imgs_04 = multicoilkdata2img_slice(padding_zero_512(data_04))
        result["num_frames"] = imgs_full.shape[0]

        # 2. Normalisation global
        v_max = np.percentile(imgs_04, 99.9) 

        if v_max > 0:
            # On divise les deux tenseurs d'un seul coup
            imgs_full = imgs_full / v_max
            imgs_04 = imgs_04 / v_max

            # On clip d'un seul coup
            imgs_full = np.clip(imgs_full, 0.0, 1.0)
            imgs_04 = np.clip(imgs_04, 0.0, 1.0)

        os.makedirs(save_full_dir, exist_ok=True)
        os.makedirs(save_04_dir, exist_ok=True)
        file_name = f"{item}_{axis_name}_s{slice_idx:02d}.npy"

        np.save(os.path.join(save_full_dir, file_name), imgs_full.astype(np.float32))
        np.save(os.path.join(save_04_dir, file_name), imgs_04.astype(np.float32))

        del raw_full, raw_mask, data_full, mask_04, imgs_full, imgs_04
        gc.collect()
        return result

    except Exception:
        result.update({"status": "ERROR", "msg": traceback.format_exc()})
        return result

def generate_slice_tasks(patient_dir: str, item: str, save_dir: str, mask_root: str, coil_info: str) -> List[Tuple]:
    tasks = []
    save_full_dir = os.path.join(save_dir, "FullSample", item)
    save_04_dir = os.path.join(save_dir, "AccFactor04", item)

    if os.path.exists(save_full_dir) and len(os.listdir(save_full_dir)) > 0:
        print(f"[SKIP]: le dossier {save_full_dir} a déjà été traité")
        return [] 

    mask_dir = os.path.join(mask_root, item)

    for axis in ["lax", "sax"]:
        full_path = os.path.join(patient_dir, f"cine_{axis}.mat")
        mask_path = os.path.join(mask_dir, f"cine_{axis}_mask_Uniform4.mat")
        
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('Mask_Task1', 'UnderSample_Task1').replace('_mask_Uniform4', '')

        if os.path.exists(full_path) and os.path.exists(mask_path):
            try:
                with h5py.File(full_path, 'r') as f:
                    num_slices = f['kspace_full'].shape[1] 
                for slice_idx in range(num_slices):
                    tasks.append((full_path, mask_path, slice_idx, axis, item, save_full_dir, save_04_dir, coil_info))
            except Exception as e:
                print(f"    [erreur] patient {item}, axe {axis} : {e}")

    if tasks: print(f"    [planification] {item} : {len(tasks)} tâches enregistrées.")
    return tasks

def generate_pairs_registry(dataset_out_dir: str) -> Tuple[int, str]:
    acc_dir = os.path.join(dataset_out_dir, "AccFactor04")
    full_dir = os.path.join(dataset_out_dir, "FullSample")
    count = 0
    if not os.path.exists(acc_dir): return 0, ""

    registry_path = os.path.join(dataset_out_dir, "pairs.txt")
    with open(registry_path, "w") as f:
        for patient in sorted(os.listdir(acc_dir)):
            p_acc = os.path.join(acc_dir, patient)
            p_full = os.path.join(full_dir, patient)
            if os.path.isdir(p_acc):
                for file in sorted(os.listdir(p_acc)):
                    f.write(f"{os.path.join(p_acc, file)} {os.path.join(p_full, file)}\n")
                    count += 1
    return count, registry_path


def main():
    parser = argparse.ArgumentParser(description="pipeline de prétraitement hpc cmrxrecon")
    parser.add_argument('-i', "--input_pre", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data_pre/home2/Raw_data/MICCAIChallenge2024/ChallengeData")
    parser.add_argument('-t', "--input_post", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data_post/GroundTruth")
    parser.add_argument('-o', "--output", type=str, default="/lustre/fsn1/projects/rech/iql/uri76kx/ig3d_CMRxRecon/data")
    parser.add_argument('-w', "--worker", type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    splits_config = [
        ('TrainingSet',   args.input_pre,  args.input_pre),
        ('ValidationSet', args.input_post, args.input_pre)
    ]

    global_tasks_queue = []
    print("--- phase 1 : analyse et planification des ressources ---")

    for split_name, img_base, mask_base in splits_config:
        full_sample_dir = os.path.join(img_base, 'MultiCoil', 'Cine', split_name, 'FullSample')
        mask_root_dir = os.path.join(mask_base, 'MultiCoil', 'Cine', split_name, 'Mask_Task1')

        print(f"\nanalyse de la partition : {split_name}...")
        for patient_item in sorted(os.listdir(full_sample_dir)):
            p_path = os.path.join(full_sample_dir, patient_item)
            if os.path.isdir(p_path) and patient_item.startswith('P'):
                global_tasks_queue.extend(generate_slice_tasks(p_path, patient_item, os.path.join(args.output, split_name), mask_root_dir, 'MultiCoil'))

    if not global_tasks_queue:
        return print("\naucune nouvelle donnée identifiée pour le traitement.")

    print(f"\n--- phase 2 : exécution parallèle ({len(global_tasks_queue)} tâches, {args.worker} cpus) ---")
    
    with multiprocessing.Pool(processes=args.worker) as pool:
        list(tqdm(pool.imap_unordered(process_single_slice, global_tasks_queue), total=len(global_tasks_queue)))

    print("\n--- phase 3 : consolidation des fichiers de registre ---")
    for split_name, _, _ in splits_config:
        split_dir = os.path.join(args.output, split_name)
        if os.path.isdir(split_dir):
            n, p = generate_pairs_registry(split_dir)
            if n > 0: print(f"  [{split_name}] registre généré : {p} ({n} paires)")


if __name__ == "__main__":
    main()
