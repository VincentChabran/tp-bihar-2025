from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random

from transforms_image import get_image_classification_transforms

def get_filtered_dataloader(
    root_dir: str,
    selected_classes: list,
    max_images_per_class: int = None,
    image_size: tuple = (128, 128),
    batch_size: int = 16,
    shuffle: bool = True
):
    """
    Charge un DataLoader ImageFolder filtré sur certaines classes et limité en taille.

    Args:
        root_dir (str): Dossier racine contenant les sous-dossiers par classe
        selected_classes (list): Liste des noms de classes à garder
        max_images_per_class (int): Limite par classe (None = toutes)
        image_size (tuple): Taille d'image à redimensionner
        batch_size (int): Taille des batchs
        shuffle (bool): Mélanger les données ?

    Returns:
        DataLoader prêt à l'emploi
    """
    transform = get_image_classification_transforms(image_size=image_size)
    full_dataset = ImageFolder(root=root_dir, transform=transform)

    # Mapping classe -> indices
    class_to_idx = full_dataset.class_to_idx
    selected_indices = []

    for class_name in selected_classes:
        class_idx = class_to_idx[class_name]
        indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label == class_idx]
        if max_images_per_class:
            indices = random.sample(indices, min(max_images_per_class, len(indices)))
        selected_indices.extend(indices)

    # Sous-dataset réduit
    reduced_dataset = Subset(full_dataset, selected_indices)
    loader = DataLoader(reduced_dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, [c for c in class_to_idx if c in selected_classes]
