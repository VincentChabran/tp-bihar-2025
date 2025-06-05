import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_vgg(num_classes: int, feature_extract: bool = True):
    """
    Charge VGG16 préentraîné et adapte la dernière couche pour num_classes.

    Args:
        num_classes (int): Nombre de classes de sortie
        feature_extract (bool): Si True, on fige tous les poids sauf le classifier

    Returns:
        torch.nn.Module: modèle VGG16 adapté
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Geler les poids si extraction de features
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # Remplacer la dernière couche du classifier
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model
