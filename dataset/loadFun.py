import numpy as np
import scipy.io as sio
import h5py
import scipy.fft as sp_fft



def loadmat(file_path):
    """
    Lit un fichier .mat de manière robuste (v7.3 via h5py).
    """
    try:
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for k, v in f.items():
                mat_data[k] = np.array(v)
        return mat_data
    except Exception as e:
        raise RuntimeError(f"Impossible de lire le fichier {file_path} : {str(e)}")


def multicoilkdata2img(kdata):
    """
    Transforme l'espace K multi-antennes en images spatiales réelles.
    Désactivation du multithreading interne (workers=1) pour éviter
    l'oversubscription avec le multiprocessing global.
    """
    # 1. IFFT 2D sur les deux derniers axes (Hauteur, Largeur)
    img_complex = sp_fft.ifftshift(
        sp_fft.ifft2(
            sp_fft.ifftshift(kdata, axes=(-2, -1)), 
            axes=(-2, -1),
            workers=1
        ),
        axes=(-2, -1)
    )

    # Combinaison des bobines par "Root Sum of Squares" (RSS)
    img_rss = np.sqrt(np.sum(np.abs(img_complex)**2, axis=-3)).astype(np.float32)
    
    return img_rss
