import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_embed_similarity(
    embeds,
    perfs,
    ordered=True,
    title="Code Embedding Cosine Similarity",
    fig=None,
    axs=None,
    vmin=None,
    vmax=None,
):
    """
    Plot the similarity of embeddings and the performance of programs.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(embeds)

    if ordered:
        from scipy.cluster.hierarchy import linkage, leaves_list

        # Perform hierarchical clustering
        linkage_matrix = linkage(embeds, method="ward")
        ordered_indices = leaves_list(linkage_matrix)

        # Reorder matrix
        similarity_matrix = similarity_matrix[ordered_indices][:, ordered_indices]
        perfs = perfs[ordered_indices]
        title += " (Clustered)"

    # Plot similarity matrix
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [20, 1]}
    )
    sns.heatmap(similarity_matrix, cmap="viridis", ax=axs[0])
    axs[0].set_title(title, fontsize=25)
    axs[0].set_xlabel("Program Index")
    axs[0].set_ylabel("Program Index")

    if ordered:
        # set xticks to be the program ids using ordered_indices
        axs[0].set_xticks(np.arange(len(ordered_indices))[::3])
        axs[0].set_xticklabels(ordered_indices[::3])
        axs[0].set_yticks(np.arange(len(ordered_indices))[::3])
        axs[0].set_yticklabels(ordered_indices[::3])

    # Plot performance heatmap
    sns.heatmap(
        perfs.reshape(-1, 1),
        cmap="Reds_r",
        ax=axs[1],
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
    )
    axs[1].set_title("Score", fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.tight_layout()
    return fig, axs, similarity_matrix
