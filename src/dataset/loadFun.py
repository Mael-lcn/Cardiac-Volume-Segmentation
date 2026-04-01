import numpy as np
import h5py
import scipy.fft as sp_fft
from typing import Optional



def load_h5_slice(file_path: str, slice_idx: int, dataset_name: Optional[str] = None) -> np.ndarray:
    """
    Extrait une unique tranche (slice) d'un fichier HDF5 (MATLAB v7.3) via lecture paresseuse (Lazy Loading).
    
    Cette fonction empêche le chargement de l'intégralité du tenseur K-space en mémoire (OOM).
    Elle lit directement sur le disque dur les octets correspondants à l'index demandé.

    Args:
        file_path (str): Chemin absolu vers le fichier .mat.
        slice_idx (int): L'index de la tranche temporelle/spatiale à extraire.
        dataset_name (str, optional): Nom de la clé HDF5 à lire (ex: 'kspace_full'). 
                                      Si None, la première clé non-système est utilisée.

    Returns:
        np.ndarray: Le tenseur numpy de la tranche isolée, de forme (Frames, Coils, H, W).

    Raises:
        KeyError: Si la clé demandée est introuvable dans le fichier.
        RuntimeError: Si le fichier est corrompu ou illisible.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name is None:
                # Auto-détection de la première vraie matrice
                dataset_name = next((k for k in f.keys() if not k.startswith('__')), None)

            if dataset_name not in f:
                raise KeyError(f"La clé '{dataset_name}' est absente du fichier {file_path}.")

            dataset = f[dataset_name]
            ndim = len(dataset.shape)

            # Si c'est le K-space (5D : frames, slices, coils, y, x)
            if ndim == 5:
                return dataset[:, slice_idx, :, :, :]
            # Si c'est le masque (généralement 2D [y, x] ou 3D), on le charge en entier
            else:
                return dataset[:]

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture paresseuse de {file_path} : {str(e)}")


def multicoilkdata2img_slice(kdata_slice: np.ndarray) -> np.ndarray:
    """
    Transforme un tenseur K-space multi-antennes (d'une seule tranche) en image spatiale réelle.
    
    Utilise la transformée de Fourier rapide (IFFT) de Scipy et combine les bobines
    via la méthode Root Sum of Squares (RSS).

    Args:
        kdata_slice (np.ndarray): Tenseur complexe de l'espace K, de forme (Frames, Coils, H, W).

    Returns:
        np.ndarray: Tenseur d'image réelle normalisée en simple précision (float32), 
                    de forme (Frames, H, W).
    """
    img_coils_complex = sp_fft.fftshift(
        sp_fft.ifft2(
            sp_fft.ifftshift(kdata_slice, axes=(-2, -1)), 
            axes=(-2, -1),
            workers=1
        ),
        axes=(-2, -1)
    )

    # 2. Combinaison RSS (Root Sum of Squares) dans le domaine de l'IMAGE
    # On prend la magnitude (abs), on l'élève au carré, on somme sur les Coils (axis=1), puis racine.
    img_combined = np.sqrt(np.sum(np.abs(img_coils_complex)**2, axis=1))

    return img_combined.astype(np.float32)
