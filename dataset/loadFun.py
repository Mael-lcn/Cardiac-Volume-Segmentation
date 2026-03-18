import numpy as np
import scipy.io as sio
import h5py



def loadmat(file_path):
    """
    Lit un fichier .mat de manière robuste.
    Gère les anciens formats MATLAB (via scipy) et les nouveaux v7.3 (via h5py).
    """
    try:
        # Tente de charger avec scipy (pour les fichiers MATLAB < v7.3)
        mat_data = sio.loadmat(file_path)
        return mat_data
    except NotImplementedError:
        # Si c'est un format MATLAB récent (v7.3), scipy échoue. On utilise h5py.
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for k, v in f.items():
                # En HDF5, les matrices sont souvent transposées par rapport à MATLAB
                mat_data[k] = np.array(v).T 
        return mat_data
    except Exception as e:
        raise RuntimeError(f"Impossible de lire le fichier {file_path} : {str(e)}")


def multicoilkdata2img(kdata):
    """
    Transforme l'espace K (k-space) multi-antennes en images spatiales réelles.
    Étapes : 
    1. Transformée de Fourier Rapide Inverse (IFFT 2D)
    2. Combinaison des antennes par "Root Sum of Squares" (RSS)
    
    Shape attendue du kdata : (frames, slices, coils, Hauteur, Largeur)
    """
    # 1. IFFT 2D sur les deux derniers axes (Hauteur, Largeur)
    # L'ifftshift sert à recentrer les basses fréquences au milieu de l'image
    img_complex = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kdata, axes=(-2, -1)), 
            axes=(-2, -1)
        ), 
        axes=(-2, -1)
    )

    # 2. Combinaison des bobines (Coils)
    # Dans CMRxRecon, l'axe des bobines est généralement l'antépénultième (index -3)
    # On calcule la magnitude absolue, on la met au carré, on somme, et on prend la racine
    img_rss = np.sqrt(np.sum(np.abs(img_complex)**2, axis=-3))
    
    return img_rss
