# Privacy Risk Assessment Using Graph Neural Networks

This project implements a Graph Neural Network (GNN) model for privacy risk assessment in social networks using the Facebook ego-network dataset. The model identifies privacy risks based on network structure and node attributes, providing explainable privacy risk scores and recommendations.

## Project Overview

The project consists of several components:

1. **Data Loading and Processing**: Processes the Facebook ego-network dataset
2. **GNN Model**: Implements privacy risk assessment using Graph Neural Networks
3. **Training Pipeline**: Trains the GNN model on the Facebook dataset
4. **Evaluation and Visualization**: Evaluates privacy risks and generates visual reports

## Features

- Privacy risk scoring for individual users in a social network
- Identification of sensitive attributes contributing to privacy risks
- Network-level privacy risk assessment
- Visual representation of privacy risks in networks
- Explainable AI approach with feature attribution

## Dataset

The project uses the Facebook ego-network dataset, which contains:
- Network structure (edges between users)
- Node features (user attributes)
- Circles/communities information

Each ego-network consists of:
- `.edges` files: Edge lists representing connections
- `.feat` files: Node feature matrices
- `.featnames` files: Names of features
- `.circles` files: Community memberships
- `.egofeat` files: Ego node's own features

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd privacy-risk-gnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py --data_dir facebook --model_type explainable --conv_type GAT --epochs 50
```

Command-line arguments:
- `--data_dir`: Path to the Facebook dataset directory
- `--model_type`: Type of model (`basic` or `explainable`)
- `--conv_type`: GNN layer type (`GCN`, `GAT`, or `SAGE`)
- `--hidden_dim`: Hidden dimension size
- `--num_layers`: Number of GNN layers
- `--epochs`: Number of training epochs
- `--output_dir`: Output directory for model and results

### Evaluating Privacy Risks

```bash
python evaluate.py --model_path output/best_model.pt --ego_id 0
```

Command-line arguments:
- `--model_path`: Path to the trained model
- `--ego_id`: Specific ego-network ID to evaluate (optional)
- `--model_type`: Type of model (`basic` or `explainable`)
- `--output_dir`: Output directory for results

## Model Architecture

The project implements two GNN model variants:

1. **Basic Privacy GNN**: A standard GNN model for privacy risk assessment
   - Multiple GNN layers (GCN, GAT, or GraphSAGE)
   - Structural feature incorporation
   - Privacy risk score prediction

2. **Explainable Privacy GNN**: An extension with feature attribution
   - Attention-based feature importance
   - Explanations for privacy risk scores
   - Identification of privacy-sensitive attributes

## Privacy Risk Metrics

The model assesses privacy risks based on:
- Exposure of sensitive attributes
- Network structure and connectivity
- Node degree and influence in the network
- Community/circle membership

## Output and Visualization

The evaluation produces:
- Privacy risk scores for each user
- Network visualizations with color-coded risk levels
- Feature importance analysis
- Privacy recommendations based on risk assessment
- Summary reports and statistics

## Unique Aspects

This implementation includes several unique features:
- Combined feature-based and structural approach to privacy assessment
- Explainable AI techniques for transparent privacy scoring
- Node-level and network-level risk analysis
- Integration of social network structure with attribute sensitivity
- Visual representation of privacy risks in the network

## License

This project is licensed under the MIT License - see the LICENSE file for details. 