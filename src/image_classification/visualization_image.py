
def denormalize(tensor, mean, std):
    """Inverse la normalisation ImageNet pour affichage"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


