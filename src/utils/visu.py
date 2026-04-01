import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt



def visualize_from_registry(registry_path: str, num_samples: int = 2, frame_idx: int = 2):
    """
    Lit le fichier de registre, charge les matrices Numpy et affiche les paires.
    """
    if not os.path.exists(registry_path):
        print(f"[ERREUR] Le fichier introuvable : {registry_path}")
        return

    with open(registry_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("[ERREUR] Le fichier de registre est vide.")
        return

    print(f"Chargement de {len(lines)} paires depuis {registry_path}...")

    # Sélection aléatoire pour ne pas toujours afficher la même tranche
    selected_lines = random.sample(lines, min(num_samples, len(lines)))

    for i, line in enumerate(selected_lines):
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"[AVERTISSEMENT] Ligne mal formatée ignorée : {line}")
            continue

        acc_path, full_path = parts

        try:
            # Chargement des tenseurs (format attendu : frames, H, W)
            img_acc = np.load(acc_path)
            img_full = np.load(full_path)
        except Exception as e:
            print(f"[ERREUR] Impossible de charger les fichiers : {e}")
            continue

        # Extraction de la frame demandée si le tenseur est en 3D (frames, H, W)
        # On prend la valeur absolue au cas où il resterait des composantes complexes
        if img_acc.ndim >= 3:
            # Sécurité pour ne pas dépasser l'index max des frames
            safe_frame_idx = min(frame_idx, img_acc.shape[0] - 1)
            display_acc = np.abs(img_acc[safe_frame_idx])
            display_full = np.abs(img_full[safe_frame_idx])
        else:
            display_acc = np.abs(img_acc)
            display_full = np.abs(img_full)

        # Création de la figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        file_name = os.path.basename(full_path)
        fig.suptitle(f"Échantillon {i+1}/{num_samples} - {file_name}", fontsize=14, fontweight='bold')
        
        # Affichage de l'image masquée (Entrée)
        axes[0].imshow(display_acc, cmap='gray')
        axes[0].set_title("Image Masquée (AccFactor04)")
        axes[0].axis('off')

        # Affichage de l'image cible (Vérité Terrain)
        axes[1].imshow(display_full, cmap='gray')
        axes[1].set_title("Image Post-Traitement (FullSample)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualiser des paires d'images depuis pairs.txt")
    parser.add_argument('-p', '--pairs_file', type=str, default="../../../output/in/TrainingSet/pairs.txt", 
                        help="Chemin complet vers le fichier pairs.txt")
    parser.add_argument('-n', '--num_samples', type=int, default=1, 
                        help="Nombre de paires aléatoires à afficher (défaut: 1)")
    parser.add_argument('-f', '--frame', type=int, default=0, 
                        help="Index temporel (frame) à afficher pour les images 3D (défaut: 0)")
    
    args = parser.parse_args()

    visualize_from_registry(args.pairs_file, args.num_samples, args.frame)