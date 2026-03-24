import os
import argparse
import numpy as np
import cv2



def normalize_to_uint8(image_sequence):
    """
    Normalise une séquence temporelle (Frames, H, W) en 8-bits (0-255).
    """
    p1, p99 = np.percentile(image_sequence, (1, 99))
    img_clipped = np.clip(image_sequence, p1, p99)
    img_normalized = (img_clipped - p1) / (p99 - p1 + 1e-8)
    return (img_normalized * 255).astype(np.uint8)


def export_sequence_to_mp4(sequence, output_filepath, fps=10):
    """
    Encode le tenseur en fichier vidéo MP4 lisible par l'interface MedSAM.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    seq_uint8 = normalize_to_uint8(sequence)
    frames, H, W = seq_uint8.shape

    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(output_filepath, fourcc, fps, (W, H))

    for idx in range(frames):
        frame = seq_uint8[idx]
        # Conversion en BGR (3 canaux) exigée par OpenCV pour la vidéo
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def main():
    parser = argparse.ArgumentParser(description="Création de vidéos MP4 en boucle pour MedSAM 2")
    parser.add_argument("--base_dir", type=str, default="../../data", help="Dossier racine des données")
    parser.add_argument("--filename", type=str, default="P004_MultiCoil_lax_all.npy", help="Nom du fichier consolidé")
    parser.add_argument("--output_dir", type=str, default="../../output/medsam_input", help="Dossier d'export")
    parser.add_argument("--cycle_length", type=int, default=12, help="Taille originale d'un battement (défaut: 12)")
    parser.add_argument("--slice_idx", type=int, default=2, help="Index de la tranche à transformer en vidéo")
    parser.add_argument("--fps", type=int, default=10, help="Vitesse de la vidéo (Images par seconde)")
    parser.add_argument("--loops", type=int, default=5, help="Nombre de fois où le cycle est répété (boucle)")
    args = parser.parse_args()

    path_full = os.path.join(args.base_dir, "FullSample", args.filename)
    path_04 = os.path.join(args.base_dir, "AccFactor04", args.filename)

    if not os.path.exists(path_full) or not os.path.exists(path_04):
        print(f"[ERREUR] Fichiers introuvables :\n -> {path_full}\n -> {path_04}")
        return

    print("Chargement des matrices...")
    data_full = np.load(path_full)
    data_04 = np.load(path_04)

    # Séparation du Temps et de la Profondeur : (Frames, Slices, H, W)
    slices = data_full.shape[0] // args.cycle_length
    data_full = data_full.reshape(args.cycle_length, slices, 512, 512)
    data_04 = data_04.reshape(args.cycle_length, slices, 512, 512)

    # Extraction d'un cycle complet pour la tranche ciblée : (Frames, H, W)
    video_full_single = data_full[:, args.slice_idx, :, :]
    video_04_single = data_04[:, args.slice_idx, :, :]

    # np.tile répète le tenseur N fois sur l'axe du temps (axe 0)
    print(f"Création d'une boucle : Répétition du cycle {args.loops} fois...")
    video_full_looped = np.tile(video_full_single, (args.loops, 1, 1))
    video_04_looped = np.tile(video_04_single, (args.loops, 1, 1))

    total_frames = video_full_looped.shape[0]

    out_full_mp4 = os.path.join(args.output_dir, "FullSample_Video.mp4")
    out_04_mp4 = os.path.join(args.output_dir, "AccFactor04_Video.mp4")

    print(f"Encodage de la vidéo Vérité Terrain ({total_frames} frames totales)...")
    export_sequence_to_mp4(video_full_looped, out_full_mp4, fps=args.fps)

    print(f"Encodage de la vidéo Accélérée 4X ({total_frames} frames totales)...")
    export_sequence_to_mp4(video_04_looped, out_04_mp4, fps=args.fps)

    print(f"\n[SUCCES] Vidéos générées dans : {args.output_dir}")
    print(f"Durée de la vidéo : {total_frames / args.fps:.1f} secondes (à {args.fps} fps).")


if __name__ == "__main__":
    main()
