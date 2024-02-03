"""
Random Walk Generator.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from functools import partial
from collections import defaultdict

import random
import networkx as nx
import logging

logger = logging.getLogger("RandomWalkGenerator")


class RandomWalk(ABC):
    """
    RandomWalk method interface definition.
    """

    def __init__(
        self,
        graph: nx.Graph,
        length: int,
        num_compressions: int,
        compression_selection_ratio: float,
    ):
        """
        Args:
            graph: Graph
            length: Random walk length
        """
        assert length >= 1, "Minimum walk length is 1!"
        if not 0 <= compression_selection_ratio <= 1:
            raise ValueError("Selection ratio must be between 0 and 1")

        self._graph = graph
        self._length = length

        self._selection_ratio = compression_selection_ratio
        self._num_compressions = num_compressions
        self._num_graphs = num_compressions + 1
        self._compressed_graphs = []
        self._hierarchical_mappings = []
        self._reverse_hierarchical_mappings = []

        for i in range(self._num_compressions):
            self.compress_graph(self._selection_ratio)

        assert len(self._compressed_graphs) == self._num_compressions
        assert len(self._compressed_graphs) == len(self._hierarchical_mappings)

    @abstractmethod
    def walk(self, node: str) -> str:
        """
        Performs a random walk starting from node `node`.
        Returns graph walk in sentence format.
        Example: `n1 n2 n3` for walk of nodes (n1, n2, n3)

        Args:
            node: Starting node

        Returns:
            walk as a sentence.
        """
        pass

    def get_graph(self, idx=0) -> nx.Graph:
        if idx == 0:
            return self._graph
        else:
            # NOTE: we consider the original graph as idx=0, so please subtract 1 from the index to get the location in the lists
            assert (
                len(self._compressed_graphs) > idx - 1
            ), f"Attempting to access a compressed graph at level {idx-1}, but does not exist"
            return self._compressed_graphs[idx - 1]

    def get_most_compressed_graph(self) -> nx.Graph:
        return self.get_graph(len(self._compressed_graphs))

    def get_node_neighbors(self, node: str, idx=0) -> List[str]:
        # When we compress nodes, we keep track of which node it was compressed into so we can get it's neighbors in the compressed graph
        mapped_node = node
        for mapping_idx in range(idx):
            assert (
                mapped_node in self._hierarchical_mappings[mapping_idx]
            ), f"Mapped node should exist in mapping: {mapping_idx=}"
            mapped_node = self._hierarchical_mappings[mapping_idx][mapped_node]

        assert mapped_node in self.get_graph(
            idx
        ), f"Mapped Node should be in the compressed graph, otherwise the mappings are wrong, {mapped_node=}, {mapping_idx=}"
        # if node not in self.get_graph(idx):
        #     node = orig_node
        #     for mapping_idx in range(idx):
        #         print(f"{mapping_idx=} / {idx=}")
        #         print(f"{node=}")
        #         node = self._hierarchical_mappings[mapping_idx][node]

        mapped_neighbors = list(self.get_graph(idx).neighbors(mapped_node))

        # Now we need to walk back down the mappings to get the node in the original graph
        # This prevents nodes that were not compressed from being skewed upwards in weight in the embedding
        out = []
        for mapped_neighbor in mapped_neighbors:
            for mapping_idx in range(idx - 1, -1, -1):
                mapped_neighbor = random.choice(
                    self._reverse_hierarchical_mappings[mapping_idx][mapped_neighbor]
                )
            out.append(mapped_neighbor)
        return out

    def get_node_unnormalized_edge_weights(self, node: str, idx=0) -> List[float]:
        neighbors = self.get_node_neighbors(node, idx)
        if not nx.is_weighted(self.get_graph(idx)):
            return [1 for _ in neighbors]
        return [self.get_graph(idx)[node][neighbor]["weight"] for neighbor in neighbors]

    def get_node_normalized_edge_weights(self, node: str, idx: int) -> List[float]:
        neighbor_weights = self.get_node_unnormalized_edge_weights(node, idx)
        neighbor_weight_sum = sum(neighbor_weights)
        return [nw / neighbor_weight_sum for nw in neighbor_weights]

    def compress_graph(self, compression_selection_ratio):
        graph = self.get_most_compressed_graph()

        nodes = list(graph.nodes())
        num_to_select = int(len(nodes) * compression_selection_ratio)
        selected_nodes = set(random.sample(nodes, num_to_select))

        compressed_graph = graph.copy()

        # Mapping from original to compressed nodes
        node_mapping = {node: node for node in graph.nodes()}
        reverse_mapping = defaultdict(lambda: [])
        for node in selected_nodes:
            if node not in compressed_graph:
                continue

            neighbors = list(compressed_graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor == node:
                    continue
                # Transfer edges from neighbor to current node
                for neighbor_neighbor in compressed_graph.neighbors(neighbor):
                    if neighbor_neighbor != node:
                        compressed_graph.add_edge(node, neighbor_neighbor)

                node_mapping[neighbor] = node

                # Deals with when neighbors are nodes that have already been compressed,
                # so their mapping must be updated
                for key, value in node_mapping.items():
                    if value == neighbor:
                        node_mapping[key] = node

                compressed_graph.remove_node(neighbor)

        for k, v in node_mapping.items():
            reverse_mapping[v].append(k)

        # A quick check
        for orig_node, mapped_node in node_mapping.items():
            assert (
                mapped_node in compressed_graph
            ), "mapped node should be in the compressed graph"

        print(
            "edge,node",
            compressed_graph.number_of_edges(),
            compressed_graph.number_of_nodes(),
        )
        self._compressed_graphs.append(compressed_graph)
        self._hierarchical_mappings.append(node_mapping)
        self._reverse_hierarchical_mappings.append(reverse_mapping)


class AdversarialRandomWalk(RandomWalk):
    """
    RandomWalk method interface definition.
    """

    def __init__(
        self,
        graph: nx.Graph,
        length: int,
        num_compressions: int,
        compression_selection_ratio: float,
        prior_transformation,
        num_perturbations_prior,
        step_transformation,
        num_perturbations_step,
        **kwargs,
    ):
        """
        Args:
            graph: Graph
            length: Random walk length
        """
        assert length >= 1, "Minimum walk length is 1!"
        if not 0 <= compression_selection_ratio <= 1:
            raise ValueError("Selection ratio must be between 0 and 1")

        self._graph = graph
        self._length = length

        self._selection_ratio = compression_selection_ratio
        self._num_compressions = num_compressions
        self._num_graphs = num_compressions + 1
        self._compressed_graphs = []
        self._hierarchical_mappings = []
        self._reverse_hierarchical_mappings = []
        self.prior_transformation = partial(
            prior_transformation, k=num_perturbations_prior
        )
        self.step_transformation = partial(
            step_transformation, k=num_perturbations_step
        )
        self.prior_transformation(self._graph)
        for i in range(self._num_compressions):
            self.compress_graph(self._selection_ratio)

        assert len(self._compressed_graphs) == self._num_compressions
        assert len(self._compressed_graphs) == len(self._hierarchical_mappings)


class DeepWalk(RandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1403.6652.pdf
    """

    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]
        graph_idx = random.choices(
            [i for i in range(self._num_graphs)],
            [1 * pow(self._selection_ratio, i) for i in range(self._num_graphs)],
            k=1,
        )[0]
        node = walk_nodes[0]
        for _ in range(self._length):
            neighbors = self.get_node_neighbors(node, graph_idx)
            if len(neighbors) < 1:
                walk_nodes.append(node)
                continue
            normalized_weights = self.get_node_normalized_edge_weights(node, graph_idx)
            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)
            node = child

        return " ".join(walk_nodes)


class AdversarialDeepWalk(AdversarialRandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1403.6652.pdf
    """

    def walk(
        self,
        node: str,
    ) -> str:
        walk_nodes: List[str] = [node]
        graph_idx = random.choices(
            [i for i in range(self._num_graphs)],
            [1 * pow(self._selection_ratio, i) for i in range(self._num_graphs)],
            k=1,
        )[0]

        node = walk_nodes[0]
        for _ in range(self._length):
            neighbors = self.get_node_neighbors(node, graph_idx)
            if len(neighbors) < 1:
                walk_nodes.append(node)
                continue
            normalized_weights = self.get_node_normalized_edge_weights(node, graph_idx)
            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)
            node = child
            self.step_transformation(self._graph)

        return " ".join(walk_nodes)


class Node2Vec(RandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1607.00653.pdf
    """

    def __init__(
        self,
        graph: nx.Graph,
        length: int,
        num_compressions: int,
        compression_selection_ratio: float,
        p: float = 1.0,
        q: float = 1.0,
    ):
        """
        Args:
            graph: Graph
            length: Random walk length
            p: Parameter p
        """
        super().__init__(
            graph=graph,
            length=length,
            num_compressions=num_compressions,
            compression_selection_ratio=compression_selection_ratio,
        )
        self._p = p
        self._q = q

    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]
        graph_idx = random.choices(
            [i for i in range(self._num_graphs)],
            [1 * pow(self._selection_ratio, i) for i in range(self._num_graphs)],
            k=1,
        )[0]
        prev_node = None
        node = walk_nodes[0]
        for _ in range(self._length):
            neighbors = self.get_node_neighbors(node, graph_idx)
            if len(neighbors) < 1:
                walk_nodes.append(node)
                continue
            neighbor_weights = self.get_node_unnormalized_edge_weights(node, graph_idx)
            for i, candidate_child in enumerate(neighbors):
                if (
                    candidate_child == prev_node
                ):  # shortest path length between "next" and previous node is 0
                    neighbor_weights[i] *= 1 / self._p
                    continue

                candidate_neighbors = self.get_node_neighbors(candidate_child)
                if (
                    prev_node in candidate_neighbors
                ):  # shortest path length between "next" and previous node is 1
                    neighbor_weights[i] *= 1 / self._q

            neighbor_weight_sum = sum(neighbor_weights)
            normalized_weights = [nw / neighbor_weight_sum for nw in neighbor_weights]

            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)

            prev_node = node
            node = child

        return " ".join(walk_nodes)


class AdversarialNode2Vec(AdversarialRandomWalk):
    """
    Implementation of simple random walk generator.
    Reference: https://arxiv.org/pdf/1607.00653.pdf
    """

    def __init__(
        self,
        graph: nx.Graph,
        length: int,
        num_compressions: int,
        compression_selection_ratio: float,
        prior_transformation,
        num_perturbations_prior,
        step_transformation,
        num_perturbations_step,
        p: float = 1.0,
        q: float = 1.0,
    ):
        """
        Args:
            graph: Graph
            length: Random walk length
            p: Parameter p
        """
        super().__init__(
            graph,
            length,
            num_compressions,
            compression_selection_ratio,
            prior_transformation,
            num_perturbations_prior,
            step_transformation,
            num_perturbations_step,
        )
        self._p = p
        self._q = q

    def walk(self, node: str) -> str:
        walk_nodes: List[str] = [node]
        prev_node = None
        graph_idx = random.choices(
            [i for i in range(self._num_graphs)],
            [1 * pow(self._selection_ratio, i) for i in range(self._num_graphs)],
            k=1,
        )[0]
        node = walk_nodes[0]
        for _ in range(self._length):
            neighbors = self.get_node_neighbors(node, graph_idx)
            if len(neighbors) < 1:
                walk_nodes.append(node)
                continue
            neighbor_weights = self.get_node_unnormalized_edge_weights(node, graph_idx)
            for i, candidate_child in enumerate(neighbors):
                if (
                    candidate_child == prev_node
                ):  # shortest path length between "next" and previous node is 0
                    neighbor_weights[i] *= 1 / self._p
                    continue

                candidate_neighbors = self.get_node_neighbors(candidate_child)
                if (
                    prev_node in candidate_neighbors
                ):  # shortest path length between "next" and previous node is 1
                    neighbor_weights[i] *= 1 / self._q

            neighbor_weight_sum = sum(neighbor_weights)
            normalized_weights = [nw / neighbor_weight_sum for nw in neighbor_weights]

            child = random.choices(neighbors, weights=normalized_weights, k=1)[0]
            walk_nodes.append(child)

            prev_node = node
            node = child
            self.step_transformation(self._graph)

        return " ".join(walk_nodes)


def random_walk_factory(
    name: str,
    graph: nx.Graph,
    length: int,
    num_compressions: int,
    compression_selection_ratio: float,
    additional_params: Optional[dict] = None,
) -> RandomWalk:
    """
    Creates random walk method object.

    Args:
        name: Method name
        graph: Graph
        length: Walk length
        additional_params: Additional method specific parameters
    Returns:
        RandomWalk generator.
    """
    name = name.lower()
    if additional_params is None:
        additional_params = {}
    SUPPORTED_METHODS = {
        "deepwalk": DeepWalk,
        "dfs": DeepWalk,
        "node2vec": Node2Vec,
        "adv_deepwalk": AdversarialDeepWalk,
        "adv_node2vec": AdversarialNode2Vec,
    }
    # print(additional_params)
    SUPPORTED_PERTUBATIONS = {
        "randomly_add_edges": randomly_add_edges,
        "randomly_remove_edges": randomly_remove_edges,
        "randomly_add_nodes": randomly_add_nodes,
        "remove_nodes_degree_centrality": remove_nodes_degree_centrality,
        "remove_nodes_betweeness_centrality": remove_nodes_betweeness_centrality,
        "remove_bridges": remove_bridges,
        "do_nothing": do_nothing,
    }
    if "prior_transformation" in additional_params:
        additional_params["prior_transformation"] = SUPPORTED_PERTUBATIONS[
            additional_params["prior_transformation"]
        ]
    if "step_transformation" in additional_params:
        additional_params["step_transformation"] = SUPPORTED_PERTUBATIONS[
            additional_params["step_transformation"]
        ]
    assert (
        name in SUPPORTED_METHODS
    ), f'Unknown method "{name}". Supported: {list(SUPPORTED_METHODS.keys())}'

    return SUPPORTED_METHODS[name](
        graph=graph,
        length=length,
        num_compressions=num_compressions,
        compression_selection_ratio=compression_selection_ratio,
        **additional_params,
    )


def randomly_add_edges(graph: nx.Graph, k):
    nonedges = list(nx.non_edges(graph))
    random.shuffle(nonedges)
    for i in range(min(k, len(nonedges))):
        if nx.is_weighted(graph):
            w = graph.size(weight="weight") // graph.size()
            graph.add_edge(
                nonedges[i][0],
                nonedges[i][1],
                weight=random.randint(0, w) + random.random(),
            )
        else:
            graph.add_edge(nonedges[i][0], nonedges[i][1])


def randomly_remove_edges(graph: nx.Graph, k):
    edges_to_remove = random.sample(list(graph.edges()), k)
    graph.remove_edges_from(edges_to_remove)


def randomly_add_nodes(graph: nx.Graph, k):
    graph.add_nodes_from([f"random{i}" for i in range(k)])
    nonedges = list(nx.non_edges(graph))
    random.shuffle(nonedges)
    edges_to_add = random.randint(k * graph.size() // 2, k * graph.size())
    for i in range(min(edges_to_add, len(nonedges))):
        if nx.is_weighted(graph):
            w = graph.size(weight="weight") // graph.size()
            graph.add_edge(
                nonedges[i][0],
                nonedges[i][1],
                weight=random.randint(0, w) + random.random(),
            )
        else:
            graph.add_edge(nonedges[i][0], nonedges[i][1])

    logger.info("Adding %d nodes", k)


def remove_nodes_degree_centrality(graph: nx.Graph, k):
    nodes_to_remove = list(
        zip(*sorted(list(nx.degree_centrality(graph).items()), key=lambda x: -x[1]))
    )[0][:k]
    graph.remove_nodes_from(nodes_to_remove)
    logger.info("Removing %d nodes: %s", k, nodes_to_remove)
    return graph


def remove_nodes_betweeness_centrality(graph: nx.Graph, k):
    nodes_to_remove = list(
        zip(
            *sorted(list(nx.betweenness_centrality(graph).items()), key=lambda x: -x[1])
        )
    )[0][:k]
    graph.remove_nodes_from(nodes_to_remove)
    logger.info("Removing %d nodes: %s", k, nodes_to_remove)
    return graph


def remove_bridges(graph: nx.Graph, k):
    edges_to_remove = list(nx.bridges(graph))[:k]
    graph.remove_edges_from(edges_to_remove)


def do_nothing(graph: nx.Graph, k):
    return
