import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def main():
    parser = argparse.ArgumentParser(description="Générateur de visuels cool pour CMRxRecon")
    parser.add_argument("--base_dir", type=str, default="../../data", help="Dossier contenant FullSample et AccFactor04")
    parser.add_argument("--filename", type=str, default="P004_MultiCoil_lax_all.npy", help="Nom exact du fichier")
    parser.add_argument("--output_dir", type=str, default="../../output", help="Dossier de destination")
    parser.add_argument("--frames", type=int, default=12, help="Nombre de frames temporelles (défaut: 12)")
    parser.add_argument("--slice_idx", type=int, default=1, help="Index de la coupe pour la vidéo et l'erreur (défaut: 1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    path_full = os.path.join(args.base_dir, "FullSample", args.filename)
    path_04 = os.path.join(args.base_dir, "AccFactor04", args.filename)

    if not os.path.exists(path_full) or not os.path.exists(path_04):
        print(f"[ERREUR] Fichiers introuvables :\n -> {path_full}\n -> {path_04}")
        return

    data_full = np.load(path_full)
    data_04 = np.load(path_04)

    # Restructuration : (frames, slices, 512, 512)
    slices = data_full.shape[0] // args.frames
    data_full = data_full.reshape(args.frames, slices, 512, 512)
    data_04 = data_04.reshape(args.frames, slices, 512, 512)

    # Rognage (Crop) pour centrer le coeur
    crop_size = 200
    center = 256
    c_min, c_max = center - crop_size//2, center + crop_size//2

    path_static = os.path.join(args.output_dir, "probleme_ia_erreur.png")
    path_gif = os.path.join(args.output_dir, "coeur_battant.gif")
    path_panorama = os.path.join(args.output_dir, "panorama_profondeur.png")

    print("1/3 Génération de la carte d'erreur (Vue statique)...")
    create_static_analysis(data_full[0, args.slice_idx], data_04[0, args.slice_idx], c_min, c_max, path_static)

    print("2/3 Génération du GIF (La dynamique temporelle)...")
    create_gif_animation(data_full[:, args.slice_idx], data_04[:, args.slice_idx], c_min, c_max, path_gif)

    print(f"3/3 Génération du Panorama 3D ({slices} tranches détectées)...")
    # On passe toutes les tranches (slices) pour la frame 0
    create_slice_panorama(data_full[0], data_04[0], c_min, c_max, path_panorama)

    print(f"\n[SUCCES] 3 Visuels générés dans '{args.output_dir}' :")
    print(f" -> {path_static}")
    print(f" -> {path_gif}")
    print(f" -> {path_panorama}")


def create_static_analysis(img_full, img_04, c_min, c_max, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    fig.suptitle("La Perte d'Information : Pourquoi l'IA est nécessaire", color='white', fontsize=16, fontweight='bold')

    img_full_crop = img_full[c_min:c_max, c_min:c_max]
    img_04_crop = img_04[c_min:c_max, c_min:c_max]
    error_map = np.abs(img_full_crop - img_04_crop)

    axes[0].imshow(img_full_crop, cmap='gray')
    axes[0].set_title("Vérité Terrain (Cible)", color='white', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(img_04_crop, cmap='gray')
    axes[1].set_title("Accélération 4X (Entrée)", color='white', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(error_map, cmap='magma', vmax=np.max(error_map)*0.8)
    axes[2].set_title("Carte des Artéfacts à corriger", color='white', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close()


def create_gif_animation(seq_full, seq_04, c_min, c_max, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    fig.suptitle("Dynamique Cardiaque dans le Temps (IRM Ciné)", color='white', fontsize=14)

    axes[0].set_title("Vérité Terrain", color='white', fontsize=12)
    axes[1].set_title("Donnée Bruitée (4X)", color='white', fontsize=12)
    axes[0].axis('off')
    axes[1].axis('off')

    im_full = axes[0].imshow(seq_full[0, c_min:c_max, c_min:c_max], cmap='gray')
    im_04 = axes[1].imshow(seq_04[0, c_min:c_max, c_min:c_max], cmap='gray')

    plt.tight_layout()

    def update(frame):
        im_full.set_array(seq_full[frame, c_min:c_max, c_min:c_max])
        im_04.set_array(seq_04[frame, c_min:c_max, c_min:c_max])
        return [im_full, im_04]

    ani = animation.FuncAnimation(fig, update, frames=seq_full.shape[0], interval=100, blit=True)
    ani.save(save_path, writer='pillow', fps=10, savefig_kwargs={'facecolor':'black'})
    plt.close()


def create_slice_panorama(volume_full, volume_04, c_min, c_max, save_path):
    """
    Crée une grille montrant l'évolution de la profondeur (Slices) à un instant T.
    """
    num_slices = volume_full.shape[0]
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8), facecolor='black')
    fig.suptitle("Exploration Volumétrique (Profondeur du Cœur)", color='white', fontsize=18, fontweight='bold')

    # Si on n'a qu'une seule tranche (peu probable, mais sécurité), on force axes à être 2D
    if num_slices == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(num_slices):
        img_f = volume_full[i, c_min:c_max, c_min:c_max]
        img_4 = volume_04[i, c_min:c_max, c_min:c_max]

        # Ligne du haut : FullSample
        axes[0, i].imshow(img_f, cmap='gray')
        axes[0, i].set_title(f"Tranche {i+1} (Vérité)", color='white', fontsize=12)
        axes[0, i].axis('off')

        # Ligne du bas : 4X
        axes[1, i].imshow(img_4, cmap='gray')
        axes[1, i].set_title(f"Tranche {i+1} (4X)", color='white', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
