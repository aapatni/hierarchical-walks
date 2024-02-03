# Hierarchical Node Embeddings for Robust Graph Machine Learning
Course project for CS 8803 MLG
Prof: Yunan Luo @ Georgia Tech

## Overview
This project investigates the robustness of graph embedding techniques against adversarial network perturbations, introducing a novel approach: hierarchical node embeddings. By integrating information from the original graph and its coarsened versions obtained through hierarchical clustering, we aim to enhance embedding robustness to disturbances. Our methodology includes simulating adversarial perturbations and assessing the embeddings' quality and robustness using node classification and link prediction tasks.

## Key Contributions
- Analysis of the robustness of random-walk-based embeddings to adversarial perturbations.
- Proposal and evaluation of hierarchical node embeddings to improve global context and robustness.
- Experimental validation on the Cora dataset, demonstrating the potential of our method to maintain or improve performance under adversarial conditions.

## Methodology
1. **Perturbation**: Simulate adversarial perturbations through methods like edge addition/removal and node creation/removal.
2. **Embedding Generation**: Utilize DeepWalk/Node2Vec paradigms, enhancing them with hierarchical clustering for a global context.
3. **Evaluation**: Use logistic regression for node and link prediction tasks, focusing on classification accuracy as the main performance indicator.

## Future Directions
Further research will explore:
- Robustness of embeddings in heterogeneous networks.
- Modifications to random-walk-based embedding schemes for additional global context.
- Evaluation across a variety of datasets (CORA/PPI) to generalize the findings.

## Contributions
- **Manoj Niverthi**: Focused on evaluating the robustness and quality of node embeddings.
- **Adam Patni**: Worked on hierarchical clustering and compression.

## How to Use
Details on how to replicate our experiments and use our hierarchical node embedding technique are provided in the accompanying code and documentation.

1. Set up conda env
2. Set up config for the experiment you'd like to run, this includes choices like how much you'd like to perturb the graph adversarially, how large your embedding dimension will be, and the number of graph coarsening steps you'd like to take. See `hw/configs/*` for examples of how to set up a config. Remember that we use the hydra/omegaconf toolset to manage dynamically building configs.
3. Run `hw/tools/download_dataset.sh` to retrieve relevant dataset
4. Run `hw/tools/train.py` to generate embeddings and train on downstream task
5. Run `hw/tools/model_analysis.py` to visualize and compute accuracy on downstream task

## References
For a detailed list of references and inspirations for this work, including foundational papers on graph embeddings and adversarial network perturbations, see the References section of our paper.

Code in this repo is heavily inspired by ![prior work](https://github.com/Robotmurlock/Deepwalk-and-Node2vec)

