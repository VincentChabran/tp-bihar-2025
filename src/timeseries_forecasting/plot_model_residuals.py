import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_residuals(residuals_dict):
    """
    Affiche la distribution des résidus pour plusieurs modèles.

    Args:
        residuals_dict (dict): Dictionnaire où chaque clé est un nom de modèle,
                               et chaque valeur est un dict avec la clé "residuals" (série de résidus).
    """
    plt.figure(figsize=(14, 6))
    
    for model_name, data in residuals_dict.items():
        residuals = data["residuals"]
        sns.kdeplot(residuals, label=model_name, fill=True, alpha=0.3)

    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Distribution des résidus par modèle")
    plt.xlabel("Résidu")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
