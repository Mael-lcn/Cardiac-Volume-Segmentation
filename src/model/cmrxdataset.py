import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset



class CMRxReconDataset(Dataset):
    """
    Dataset personnalisé pour CMRxRecon supportant plusieurs résolutions.
    """
    def __init__(self, mapping_file, image_size=(128, 128), transform=None):
        """
        Args:
            mapping_file (str): Chemin vers le fichier pairs.txt généré par data_task.py.
            image_size (tuple): Résolution cible (Hauteur, Largeur).
            transform (callable, optional): Transformations PyTorch supplémentaires.
        """
        self.pairs = []
        with open(mapping_file, 'r') as f:
            for line in f:
                self.pairs.append(line.strip().split())
        
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Chemins vers les fichiers .npy (sauvegardés en 512x512 par data_task.py)
        input_path, target_path = self.pairs[idx]

        # Chargement des tenseurs (frames, 512, 512)
        # On utilise np.load car data_task.py sauvegarde au format numpy
        input_data = np.load(input_path)
        target_data = np.load(target_path)

        # Conversion en tenseurs PyTorch (Frames, 1, H, W)
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(1)
        target_tensor = torch.from_numpy(target_data).float().unsqueeze(1)

        # Redimensionnement dynamique à la taille spécifiée
        input_tensor = TF.resize(input_tensor, self.image_size, antialias=True)
        target_tensor = TF.resize(target_tensor, self.image_size, antialias=True)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor
