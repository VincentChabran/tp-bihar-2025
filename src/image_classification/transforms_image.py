from torchvision import transforms

def get_image_classification_transforms(image_size=(128, 128), normalize=True):
    """
    Transformations pour la classification d'images.
    - Redimensionne
    - Convertit en tensor
    - Normalise (optionnel)
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]

    if normalize:
        # Normalisation ImageNet (valeurs standard pour modèles préentraînés)
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)
