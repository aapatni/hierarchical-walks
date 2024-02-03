# Hierarchical Node Embeddings for Robust Graph Machine Learning
Course project for CS 8803 Machine Learning for Graphs

Adam Patni & Manoj Niverthi

Prof: Yunan Luo @ Georgia Tech

## Overview
This project investigates the robustness of graph embedding techniques against adversarial network perturbations, introducing a novel approach: hierarchical node embeddings. By integrating information from the original graph and its coarsened versions obtained through hierarchical clustering, we aim to enhance embedding robustness to disturbances. Our methodology includes simulating adversarial perturbations and assessing the embeddings' quality and robustness using node classification and link prediction tasks.

Below, you can visualize the embedding perturbation and hierarchicalization processes.
<img width="400" alt="Screenshot 2024-02-03 at 9 29 19 AM" src="https://github.com/aapatni/hierarchical-walks/assets/21110240/7b47ad45-a6fc-4590-983e-1f276b90bdbc"> <img width="400" alt="Screenshot 2024-02-03 at 9 29 34 AM" src="https://github.com/aapatni/hierarchical-walks/assets/21110240/5afefabf-0c30-4e8c-963b-a8e56b320a97">

## Key Contributions
- Analysis of the robustness of random-walk-based embeddings to adversarial perturbations.
- Proposal and evaluation of hierarchical node embeddings to improve global context and robustness.
- Experimental validation on the Cora dataset, demonstrating the potential of our method to maintain or improve performance under adversarial conditions.

## Methodology
1. **Perturbation**: Simulate adversarial perturbations through methods like edge addition/removal and node creation/removal.
2. **Embedding Generation**: Utilize DeepWalk/Node2Vec paradigms, enhancing them with hierarchical clustering for a global context.
3. **Evaluation**: Use logistic regression for node and link prediction tasks, focusing on classification accuracy as the main performance indicator.


### Results

Our experiments evaluated the robustness and performance of hierarchical node embeddings compared to traditional methods like DeepWalk and Node2Vec across various adversarial perturbations. The primary focus was on node and edge classification tasks within the Cora dataset, assessing how different perturbations affect the accuracy of these embeddings.

#### Key Findings:

- The addition of nodes to the graph significantly impacts performance due to the introduction of new, uncorrelated edges that add noise to the graph, adversely affecting downstream performance.
- Performance remained relatively stable even with informed perturbations, showcasing the robustness of the embeddings against manipulations aimed at disrupting graph structure.
- Hierarchical embeddings maintained or slightly improved classification performance in most scenarios, underscoring the potential benefit of incorporating global context into the embeddings.
- Certain perturbations, notably random node addition, demonstrated that hierarchical embeddings could outperform traditional methods by leveraging additional global information.
- However, in some instances, the introduction of hierarchicalization led to a minor drop in performance, highlighting the importance of tuning hierarchical parameters to optimize performance.

#### Embedding Clustering Visualization

We perform PCA on the output embeddings across all experiments and reduce to a 2-dimensional vector, to allow for visualization. It is clear that for each experiment there are different levels of clustering visually.
<img width="817" alt="Screenshot 2024-02-03 at 9 28 39 AM" src="https://github.com/aapatni/hierarchical-walks/assets/21110240/2c8b04e2-167e-4639-b033-95fbead5106d">



#### Classification Performance:

The tables below summarize the AUCROC (Area Under the Receiver Operating Characteristic Curve) and accuracy values for node and edge classification tasks under various perturbation scenarios. These results highlight the comparative performance of Node2Vec and DeepWalk embeddings with and without hierarchical augmentation.

##### Node/Edge Classification AUCROC Values for Cora

- This table presents AUCROC values for various perturbation scenarios, comparing traditional embedding methods with and without adversarial perturbations. Then we show the impact of hierarchicalization on embedding performance, again across the same range of perturbation scenarios.

<img width="400" alt="Screenshot 2024-02-03 at 9 24 19 AM" src="https://github.com/aapatni/hierarchical-walks/assets/21110240/bfd447b2-0054-4704-ac1c-c9b5ebcc5d09"> <img width="400" alt="Screenshot 2024-02-03 at 9 24 32 AM" src="https://github.com/aapatni/hierarchical-walks/assets/21110240/25088a98-cd37-47f5-bf6d-ec5c19716a10">

These results affirm the robustness of hierarchical embeddings against adversarial perturbations and their potential to enhance graph machine learning models' performance by incorporating a global context. Further investigation into the tuning of hierarchical parameters and their application to larger or more complex datasets is necessary for future research.

For the complete analysis and a deeper dive into our findings, feel free to reach out, happy to share the full paper!

## Future Directions
Further research will explore:
- Robustness of embeddings in heterogeneous networks.
- Modifications to random-walk-based embedding schemes for additional global context.
- Evaluation across a variety of datasets (CORA/PPI) to generalize the findings.


## Usage
Details on how to replicate our experiments and use our hierarchical node embedding technique are provided in the accompanying code and documentation.

1. Set up conda env
2. Set up config for the experiment you'd like to run, this includes choices like how much you'd like to perturb the graph adversarially, how large your embedding dimension will be, and the number of graph coarsening steps you'd like to take. See `hw/configs/*` for examples of how to set up a config. Remember that we use the hydra/omegaconf toolset to manage dynamically building configs.
3. Run `hw/tools/download_dataset.sh` to retrieve relevant dataset
4. Run `hw/tools/train.py` to generate embeddings and train on downstream task
5. Run `hw/tools/model_analysis.py` to visualize and compute accuracy on downstream task

## Contributions
- **Manoj Niverthi**: Focused on evaluating the robustness and quality of node embeddings.
- **Adam Patni**: Worked on hierarchical clustering and compression.

## References
For a detailed list of references and inspirations for this work, including foundational papers on graph embeddings and adversarial network perturbations, see the References section of our paper.

Code in this repo is heavily inspired by ![prior work](https://github.com/Robotmurlock/Deepwalk-and-Node2vec)

