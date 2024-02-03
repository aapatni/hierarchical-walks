"""
Performs downstream task on dataset labels.
Note: Dataset has to support labels.
Note: This is specialized for graph shallow encoders (deepwalk/node2vec)
"""
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict, Counter

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from hw.embeddings.common.path import CONFIG_PATH
from hw.embeddings.graph import edge_operators
from hw.embeddings.split import SplitAlgorithm
from hw.embeddings.word2vec.dataloader.torch_dataset import GraphDataset
from hw.embeddings.word2vec.model import W2VBase
from hw.tools import conventions
from hw.tools.utils import setup_pipeline, MATPLOTLIB_COLORS

logger = logging.getLogger("Model Analysis")


def labels_to_integers(labels: List[str]) -> List[int]:
    """
    Convert a list of unique string labels to integers.

    Args:
        labels: List of string labels.

    Returns:
        List of integer representations of the labels.
    """
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]
    return int_labels


def visualize_class_correlations(cfg, dataset: GraphDataset):
    nodes = len(dataset.graph)
    num_labels = None

    for k, v1 in dataset.labels.items():
        if not isinstance(v1, list):
            return
        else:
            num_labels = len(v1)
            break

    corrs = np.zeros((nodes, num_labels))
    for idx1, (_, v1) in enumerate(dataset.labels.items()):
        for idx2, v in enumerate(v1):
            corrs[idx1, idx2] = int(v)
    mcc_matrix = np.zeros((num_labels, num_labels))

    for i in tqdm(range(corrs.shape[1])):
        for j in range(corrs.shape[1]):
            mcc_matrix[i, j] = matthews_corrcoef(corrs[:, i], corrs[:, j])

    # pdb.set_trace()
    heatmap = sns.heatmap(
        mcc_matrix,
        annot=False,
        cbar=True,
        square=True,
        cmap="coolwarm",
        fmt=".1f",
        vmin=-1,
        vmax=1,
    )
    heatmap.set(xlabel="Features", ylabel="Features", title="Feature-wise Correlations")
    analysis_exp_path = conventions.get_analysis_experiment_path(
        cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment
    )
    Path(analysis_exp_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(analysis_exp_path, "class_correlation.jpg")
    plt.savefig(fig_path)


def visualize_homophily(cfg, dataset: GraphDataset):
    # Create a matrix to store homophily values
    values = set(dataset.labels.values())
    num_groups = len(values)
    homophily_matrix = np.zeros((num_groups, num_groups))
    class_name_mapping = defaultdict(lambda: "")
    for idx, value in enumerate(values):
        class_name_mapping[value] = idx
    # Iterate over edges and calculate homophily
    for edge in dataset.graph.edges():
        node1, node2 = edge
        # import pdb

        # pdb.set_trace()
        group1 = class_name_mapping[dataset.labels[node1]]
        group2 = class_name_mapping[dataset.labels[node2]]

        homophily_matrix[group1][group2] += 1
        homophily_matrix[group2][group1] += 1

    # Normalize the matrix
    total_edges_per_group = np.sum(homophily_matrix, axis=1)
    homophily_matrix_normalized = (
        homophily_matrix / total_edges_per_group[:, np.newaxis]
    )
    hm = sns.heatmap(
        homophily_matrix_normalized,
        cmap="coolwarm",
        fmt=".2f",
        cbar=True,
        square=True,
        annot=True,
    )
    hm.set_xlabel("Classes")
    hm.set_ylabel("Classes")
    hm.set_title(f"Node Homophily Heatmap in {str(cfg.datamodule.dataset_name)}")
    plt.tight_layout()
    analysis_exp_path = conventions.get_analysis_experiment_path(
        cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment
    )
    Path(analysis_exp_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(analysis_exp_path, "class_homophily.png"))


def visualize_class_distribution(cfg, dataset: GraphDataset):
    counts = defaultdict(lambda: 0)
    for _, v in dataset.labels.items():
        if isinstance(v, list):
            for idx, a in enumerate(v):
                counts[idx] += int(a)
        else:
            counts[v] += 1
    names = list(counts.keys())
    values = list(counts.values())
    analysis_exp_path = conventions.get_analysis_experiment_path(
        cfg.path.output_dir, cfg.datamodule.dataset_name, cfg.train.experiment
    )
    Path(analysis_exp_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(analysis_exp_path, "class_visualization.jpg")
    # plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_title(
        f"Distribution of Class Frequencies in {str(cfg.datamodule.dataset_name)}"
    )
    ax.set_xlabel("Class Names")
    ax.set_ylabel("Number of Examples")
    fig.set_figheight(12)
    fig.set_figwidth(20)
    ax.bar(range(len(counts)), values, tick_label=names)
    fig.savefig(fig_path)
    plt.style.use("default")


@hydra.main(config_path=CONFIG_PATH, config_name="cora_dw.yaml")
def main(cfg: DictConfig) -> None:
    cfg = setup_pipeline(cfg, task="downstream-classification")
    assert cfg.datamodule.is_graph, "This script supports only graph datasets!"

    dataset: GraphDataset = cfg.datamodule.instantiate_dataset()

    visualize_class_distribution(cfg, dataset)
    visualize_class_correlations(cfg, dataset)
    visualize_homophily(cfg, dataset)


if __name__ == "__main__":
    main()
