# AI-Powered Privacy Risk Prediction for Social Media Users

This project implements a Graph Convolutional Network (GCN) to predict privacy risks for users in social networks. The system uses real Facebook ego-network data to train a model that can identify users who are at risk of having their sensitive attributes inferred through their network connections.

## Overview

### The Prediction Model (GNN: GCNPrivacyPredictor)

The GCN takes user attributes and their social network connections as input, and outputs a privacy risk score for each user. A lower score indicates higher privacy risk (easier to re-identify or infer sensitive information), while a higher score indicates better privacy.

### The Simulation Model

The simulation model applies privacy-enhancing techniques to users identified as high-risk by the GNN. It generalizes sensitive attributes for these users and recalculates privacy scores to provide new ground truth labels for the GNN to learn from.

### Iterative Learning Process

The system iteratively improves through:
1. GNN prediction of privacy risks
2. Simulation of privacy-enhancing interventions
3. Recalculation of privacy scores based on interventions
4. GNN retraining with new data and scores

## Data

The project uses the Facebook ego-network dataset provided in the `/facebook` directory, which includes:
- `.edges` files: Social connections between users
- `.feat` files: User attribute features
- `.featnames` files: Names of the features
- `.circles` files: Group membership information

## Requirements

```
numpy
pandas
torch
torch-geometric
matplotlib
networkx
scikit-learn
```

## Installation

1. Install required packages:
```bash
pip install numpy pandas torch torch-geometric matplotlib networkx scikit-learn
```

2. If you're having trouble with PyTorch Geometric, you might need to install it with specific CUDA version support:
```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```
Replace `${TORCH_VERSION}` and `${CUDA_VERSION}` with your installed versions.

## Usage

### Running the Model

To train and evaluate the model on a specific ego network:

```bash
python gnn_privacy_predictor.py
```

This will:
1. Load the Facebook data for the default ego user (ID: 0)
2. Train the GNN model iteratively with privacy simulations
3. Output performance metrics for each iteration

### Visualizing Results

To generate visualizations of the model's performance:

```bash
python visualize_results.py
```

This will produce:
1. Plots of performance metrics over iterations
2. A privacy-utility tradeoff graph
3. A network visualization with nodes colored by privacy scores
4. A distribution plot of the sensitive attribute

## Customization

You can customize the model by modifying these parameters in `gnn_privacy_predictor.py`:

- `ego_id`: ID of the ego user to analyze (default: "0")
- `num_iterations`: Number of training/simulation cycles (default: 5)
- `privacy_threshold`: Threshold below which privacy enhancement is applied (default: 0.3)
- `intervention_rate`: Proportion of at-risk users to enhance (default: 0.6)
- `sensitive_attr_idx`: Index of the feature to use as sensitive attribute (default: randomly selected)

## How It Works

1. **Data Loading**: The system loads an ego-network from the Facebook dataset.

2. **Privacy Scoring**: Initially calculates "ground truth" privacy scores based on how easily a user's sensitive attribute can be inferred from their neighbors.

3. **GNN Training**: Trains the GCN model to predict these privacy scores based on user features and network connections.

4. **Simulation**: Uses the trained model to identify high-risk users and applies privacy enhancement (attribute generalization).

5. **Iteration**: Recalculates privacy scores after enhancement and retrains the model, creating an iterative improvement cycle.

## Output

The system produces:
- Trained GNN model for privacy risk prediction
- Performance metrics (MSE, Accuracy, F1 Score, AUC)
- Privacy-utility tradeoff measurement
- Visualizations of the network and privacy scores

## License

This project is for academic and research purposes only. 