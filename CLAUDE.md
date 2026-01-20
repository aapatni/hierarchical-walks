# CLAUDE.md - AI Assistant Guide

## Project Overview

**Hierarchical Node Embeddings for Robust Graph Machine Learning**

This is a research project that investigates the robustness of graph embedding techniques against adversarial network perturbations. The project introduces hierarchical node embeddings by integrating information from the original graph and its coarsened versions obtained through hierarchical clustering.

**Key Authors:** Adam Patni & Manoj Niverthi
**Institution:** Georgia Tech (CS 8803 Machine Learning for Graphs)
**Professor:** Yunan Luo

### Research Focus
- Analysis of robustness of random-walk-based embeddings to adversarial perturbations
- Proposal and evaluation of hierarchical node embeddings for improved global context
- Experimental validation on Cora and PPI datasets
- Comparison of DeepWalk and Node2Vec with hierarchical augmentation

## Directory Structure

```
hierarchical-walks/
├── hw/                              # Main package directory
│   ├── configs/                     # Hydra configuration files
│   │   ├── base.yaml               # Base configuration template
│   │   ├── datamodule/             # Dataset-specific configs
│   │   │   ├── cora.yaml
│   │   │   └── ppi.yaml
│   │   └── *.yaml                  # Experiment-specific configs
│   ├── embeddings/                 # Core embedding logic
│   │   ├── word2vec/               # Word2Vec implementation for graphs
│   │   │   ├── model.py            # SkipGram model
│   │   │   ├── trainer.py          # PyTorch Lightning trainer
│   │   │   ├── loss.py             # Loss functions
│   │   │   ├── dataloader/         # Data loading utilities
│   │   │   └── utils/              # Helper functions
│   │   ├── graph/                  # Graph-specific modules
│   │   │   ├── random_walk_generator.py  # DeepWalk/Node2Vec walks
│   │   │   ├── datasets.py         # Graph dataset implementations
│   │   │   └── edge_operators.py   # Edge manipulation for embeddings
│   │   ├── config_parser/          # Configuration parsing utilities
│   │   ├── split/                  # Train/test split algorithms
│   │   └── common/                 # Shared utilities
│   └── tools/                      # Executable scripts
│       ├── train.py                # Training script
│       ├── model_analysis.py       # Embedding visualization & analysis
│       ├── graph_model_downstream_classification.py  # Downstream tasks
│       ├── conventions.py          # Project file structure conventions
│       ├── utils.py                # Utility functions
│       └── download_dataset.sh     # Dataset downloader
├── setup.py                        # Package installation
├── environment.yml                 # Conda environment specification
├── run_experiments.sh              # Batch experiment runner
├── run_downstream.sh               # Downstream task runner
└── run_compression_train.sh        # Compression experiments

Output Structure (generated at runtime):
{output_dir}/
├── {dataset_name}/
│   └── {experiment_name}/
│       ├── checkpoints/            # Model checkpoints
│       ├── run_history/            # Config snapshots
│       └── analysis/               # Visualization & results
└── tb_logs/                        # TensorBoard logs
    └── {dataset_name}/
        └── {experiment_name}/
```

## Core Components

### 1. Random Walk Generators (`hw/embeddings/graph/random_walk_generator.py`)

**Key Classes:**
- `RandomWalk` (ABC): Base class for random walk generation with hierarchical graph compression
- `DeepWalk`: Simple random walk implementation
- `Node2Vec`: Biased random walk with parameters p and q
- `AdversarialDeepWalk` / `AdversarialNode2Vec`: Variants with adversarial perturbations

**Hierarchical Compression:**
- Graphs are progressively coarsened through `num_compressions` iterations
- Each compression merges nodes based on `compression_selection_ratio`
- Maintains mappings between original and compressed node spaces
- Random walks can sample from different compression levels

**Adversarial Perturbations:**
- `randomly_add_edges`: Add random edges to graph
- `randomly_remove_edges`: Remove random edges
- `randomly_add_nodes`: Add new nodes with random connections
- `remove_nodes_degree_centrality`: Remove high-degree nodes
- `remove_nodes_betweeness_centrality`: Remove high-betweenness nodes
- `remove_bridges`: Remove bridge edges

### 2. Datasets (`hw/embeddings/graph/datasets.py`)

**Supported Datasets:**
- `GraphTriplets`: Simple synthetic test dataset
- `KarateClubDataset`: Classic Zachary's Karate Club
- `CoraDataset`: Citation network (2708 papers, 7 classes)
- `PPIDataset`: Protein-Protein Interaction network

**Key Properties:**
- All inherit from `RandomWalkDataset`
- Support node labels for downstream classification
- Support node features (for Cora/PPI)
- Iterate over random walks for training

### 3. Word2Vec Model (`hw/embeddings/word2vec/`)

**Implementation:**
- Uses PyTorch Lightning for training
- SkipGram architecture adapted for graph nodes
- Negative sampling for efficient training
- Input and output embeddings

**Key Files:**
- `model.py`: SkipGram model definition
- `trainer.py`: PyTorch Lightning training wrapper
- `loss.py`: Negative sampling loss
- `dataloader/`: Data iteration and batching

### 4. Configuration System (Hydra/OmegaConf)

**Configuration Hierarchy:**
```yaml
defaults:
  - base                  # Base config with common parameters
  - datamodule: cora      # Dataset-specific config

train:
  experiment: 'experiment_name'
  accelerator: 'gpu'
  devices: '1'
  max_epochs: 20

datamodule:
  dataset_name: 'graph_cora'
  walks_per_node: 8
  walk_length: 8
  num_compressions: 2              # Number of graph coarsening steps
  compression_selection_ratio: 0.5  # Ratio of nodes to keep
  additional_parameters:
    method: 'node2vec'              # or 'deepwalk', 'adv_node2vec', etc.
    method_params:
      p: 1.0                        # Node2Vec return parameter
      q: 2.0                        # Node2Vec in-out parameter
```

**Config Override Examples:**
```bash
# Override specific parameters
python hw/tools/train.py --config-name=cora_n2v train.max_epochs=50

# Override nested parameters
python hw/tools/train.py --config-name=cora_n2v \
  datamodule.additional_parameters.method_params.p=2.0
```

## Development Workflows

### Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hierarchical-walks

# Install package in development mode
pip install -e .
```

### Download Datasets

```bash
# Download specific dataset
./hw/tools/download_dataset.sh cora
./hw/tools/download_dataset.sh ppi

# Datasets are downloaded to assets/ directory
```

### Training Workflow

**1. Configure Experiment:**
Create or modify a config file in `hw/configs/`. Example pattern:
- `cora_n2v.yaml`: Standard Node2Vec on Cora
- `cora_adv_n2v_random_edge_addition.yaml`: Adversarial variant
- `cora_adv_dw_hw_*`: DeepWalk with hierarchical walks

**2. Run Training:**
```bash
python hw/tools/train.py --config-name=cora_n2v
```

The training script:
- Checks for existing experiment history (prompts to delete if exists)
- Instantiates dataset and dataloader
- Trains Word2Vec model on random walks
- Saves checkpoints to `{output_dir}/{dataset}/checkpoints/`
- Logs to TensorBoard

**3. Analyze Results:**
```bash
python hw/tools/model_analysis.py --config-name=cora_n2v
```

Analysis outputs:
- Closest word pairs (cosine similarity)
- T-SNE visualization of embeddings
- Saved to `{output_dir}/{dataset}/analysis/`

**4. Downstream Tasks:**
```bash
python hw/tools/graph_model_downstream_classification.py --config-name=cora_n2v
```

Downstream tasks:
- Node classification (logistic regression)
- Edge/link prediction
- Reports accuracy and AUCROC scores

### Batch Experiments

```bash
# Run all Cora adversarial experiments
./run_experiments.sh

# Run downstream tasks for specific pattern
./run_downstream.sh "cora_adv_*"
```

## Key Conventions

### File Naming
- **Configs:** `{dataset}_{method}_{variant}_{perturbation}.yaml`
  - Example: `cora_adv_n2v_hw_random_edge_addition.yaml`
  - `adv`: adversarial perturbations enabled
  - `hw`: hierarchical walks enabled
  - `dw`/`n2v`: DeepWalk or Node2Vec

### Code Style
- Uses `logging` module for output (not print statements)
- Type hints on function signatures
- Docstrings follow Google style
- NetworkX for graph operations
- PyTorch Lightning for model training

### Path Management
All path construction goes through `hw/tools/conventions.py`:
- `get_checkpoint_path()`: Model checkpoint location
- `get_analysis_experiment_path()`: Analysis output location
- `get_tb_logs_experiment_path()`: TensorBoard logs

**Never hardcode paths** - always use convention functions.

### Git Workflow
- Main development should occur on feature branches
- Branch naming: `claude/claude-md-{session-id}`
- Commit messages should be descriptive and reference changes
- Push to designated branch with `git push -u origin <branch-name>`

## Important Implementation Details

### Graph Compression Algorithm
Located in `RandomWalk.compress_graph()`:
1. Select `compression_selection_ratio` of nodes randomly
2. For each selected node, absorb its neighbors
3. Transfer edges from neighbors to selected node
4. Remove absorbed neighbors from graph
5. Maintain forward and reverse mappings for walk generation

### Random Walk with Hierarchy
Random walks can traverse multiple compression levels:
- At walk start, randomly select a graph level (weighted by compression ratio)
- Traverse that level's graph structure
- Map nodes back to original space for embedding

### Word2Vec Adaptation
Nodes → Words, Random Walks → Sentences
- Node IDs are treated as vocabulary tokens
- Random walks form "sentences" for training
- SkipGram predicts context nodes from target node

### Adversarial Training
Two types of perturbations:
1. **Prior transformation:** Applied once before generating any walks
2. **Step transformation:** Applied after each step in the random walk

## Testing & Validation

### Sanity Tests
- `GraphTriplets` dataset: 3 fully-connected triplets
- Should produce 3 distinct embedding clusters
- Quick validation that embeddings capture structure

### Metrics
- **Node Classification:** Accuracy, AUCROC
- **Edge Prediction:** Accuracy, AUCROC
- **Embedding Quality:** Cosine similarity, T-SNE visualization

### Expected Behavior
- DeepWalk/Node2Vec without perturbations: High accuracy baseline
- Adversarial perturbations: Degraded performance
- Hierarchical embeddings: Improved robustness to perturbations

## Common Tasks for AI Assistants

### Adding a New Dataset
1. Create dataset class in `hw/embeddings/graph/datasets.py`
2. Inherit from `RandomWalkDataset`
3. Register with `@register_dataset("graph_{name}")`
4. Load graph, labels, and features in `__init__`
5. Create config in `hw/configs/datamodule/{name}.yaml`
6. Add download script entry if needed

### Adding a New Perturbation
1. Add function to `hw/embeddings/graph/random_walk_generator.py`
2. Signature: `def perturbation_name(graph: nx.Graph, k: int) -> None`
3. Add to `SUPPORTED_PERTUBATIONS` dict in `random_walk_factory()`
4. Reference in config's `additional_parameters.prior_transformation`

### Modifying Training Parameters
Edit `hw/configs/base.yaml` or create experiment-specific config:
- `model.embedding_size`: Embedding dimensionality
- `train.max_epochs`: Training iterations
- `train.optimizer.lr`: Learning rate
- `datamodule.walks_per_node`: Random walks per node
- `datamodule.walk_length`: Steps in each walk

### Debugging Tips
1. **Check TensorBoard logs:**
   ```bash
   tensorboard --logdir {output_dir}/tb_logs --port 6006
   ```

2. **Verify dataset loading:**
   ```python
   from hw.embeddings.graph.datasets import CoraDataset
   dataset = CoraDataset(walks_per_node=1, walk_length=5,
                         num_compressions=0, compression_selection_ratio=0)
   print(f"Graph: {len(dataset.graph)} nodes, {dataset.graph.number_of_edges()} edges")
   print(f"Labels: {len(set(dataset.labels.values()))} classes")
   ```

3. **Inspect random walks:**
   ```python
   walk = next(iter(dataset))
   print(f"Walk: {walk}")
   print(f"Length: {len(walk.split())} nodes")
   ```

## Dependencies & Requirements

### Core Libraries
- **PyTorch (2.1.0):** Deep learning framework
- **PyTorch Lightning (2.1.0):** Training wrapper
- **NetworkX:** Graph operations
- **Hydra (1.3.2):** Configuration management
- **scikit-learn:** Downstream classification
- **pandas, numpy:** Data processing
- **matplotlib, seaborn:** Visualization

### Python Version
- Python 3.11 (specified in environment.yml)
- May work with 3.8+, but untested

### Hardware
- GPU strongly recommended for training (CUDA-compatible)
- CPU training is slow but functional
- Memory: 8GB+ recommended for Cora, 16GB+ for PPI

## Known Issues & Limitations

1. **Hardcoded Paths:** Some shell scripts have hardcoded paths like `~/Projects/hierarchical-walks/` - update these for your environment

2. **Dataset Availability:** PPI download link may be unstable; manual download might be required

3. **Semantic Test:** The `semantics_test` in model_analysis.py is specialized for text datasets (Shakespeare) and not applicable to graph datasets

4. **Interactive Prompts:** Training checks for existing experiments and prompts for deletion - not suitable for non-interactive environments

5. **Reproducibility:** Random seeds are not explicitly set in all places - results may vary slightly between runs

## References & Inspiration

This codebase is heavily inspired by prior work:
- [Deepwalk-and-Node2vec Repository](https://github.com/Robotmurlock/Deepwalk-and-Node2vec)

**Key Papers:**
- DeepWalk: https://arxiv.org/pdf/1403.6652.pdf
- Node2Vec: https://arxiv.org/pdf/1607.00653.pdf
- Cora Dataset: https://graphsandnetworks.com/the-cora-dataset/

## Quick Reference Commands

```bash
# Setup
conda env create -f environment.yml
conda activate hierarchical-walks
pip install -e .

# Download data
./hw/tools/download_dataset.sh cora

# Single experiment
python hw/tools/train.py --config-name=cora_n2v
python hw/tools/graph_model_downstream_classification.py --config-name=cora_n2v

# Batch experiments
./run_experiments.sh                    # All cora_adv_* configs
./run_downstream.sh "cora_adv_*"       # Downstream for pattern

# Monitoring
tensorboard --logdir runs/tb_logs --port 6006
```

## Contact & Support

For questions about the research or implementation:
- **Authors:** Adam Patni, Manoj Niverthi
- **Institution:** Georgia Tech
- **Course:** CS 8803 Machine Learning for Graphs

This is a course project - full paper available upon request to authors.
