import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizers/shared-tokenizer")

import json
from src.utils.modules import Transformer
with open(f"models/model.json") as f:
    config = json.load(f)
model = Transformer(config)
checkpoint = torch.load("models/model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

embeddings = model.shared_embeddings.token_embeddings.weight

import numpy as np
def compute_pca(embedding_matrix):
    """Reduce embedding matrix to 2D using PCA."""
    pca = PCA(n_components=2)
    return pca.fit_transform(embedding_matrix)

def plot_tokens_pca(token_groups, categories, title="Theme-based clustering"):
    """
    Plot tokens in 2D space using PCA on their embeddings.
    All tokens are projected together, but colored by their group.

    Args:
        token_groups (list of lists): List of token lists to visualize.
        title (str): Title for the plot.
    """
    # Collect all tokens and their group indices
    all_tokens = []
    all_group_indices = []
    for group_idx, tokens in enumerate(token_groups):
        for token in tokens:
            all_tokens.append(token)
            all_group_indices.append(group_idx)

    # Tokenize and get indices for all tokens
    token_indices = []
    for token in all_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if not encoded:
            print(f"Warning: Token '{token}' not found in vocabulary.")
            continue
        token_indices.append(encoded[0])

    if not token_indices:
        print("No valid tokens to plot.")
        return

    # Get embeddings for all tokens
    vecs = embeddings[token_indices].clone().detach().numpy()

    # Compute PCA on all embeddings together
    reduced = compute_pca(vecs)

    # Assign colors to each group
    unique_groups = np.unique(all_group_indices)

    # Plot
    plt.figure(figsize=(10, 8))
    for group_idx in unique_groups:
        mask = np.array(all_group_indices) == group_idx
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"{categories[group_idx]}"
        )
        # Annotate tokens for this group
        for i, token in enumerate(all_tokens):
            if all_group_indices[i] == group_idx:
                plt.annotate(
                    token,
                    (reduced[i, 0], reduced[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center"
                )

    # Add labels, title, and legend
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
france = [" fransa", " france"]
spain = [" spain", " isbanya"]
morocco_ = [" morocco", " maghrib"]
category_names = ["France", "Spain", "Morocco"]
plot_tokens_pca([france, spain, morocco_], category_names, title="Same country, different language")

food = [" zebda", " butter", " qahwa", " coffee", " atay", " tea", " khell", " vinegar", " khodra", " vegetables", " semolina", " couscous"]
countries = [" fransa", " isbanya", " france", " spain", " portugal", " paris", " states", " senegal"]
morocco = [" maghrib", " morocco", " tanja", " lbeida", " marrakech", " essaouira", " rabat", " warzazat", " casablanca", " fes", " meknes", " agadir"]
numbers = [" wahed", " jouj", " tlata", " rba", " setta", " tnin", " tmanya"]
all = [food, countries, morocco, numbers]
category_names = ["Food", "Countries", "Moroccan Cities", "Numbers"]

plot_tokens_pca(all, category_names)