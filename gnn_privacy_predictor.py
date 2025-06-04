import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class GCNPrivacyPredictor(nn.Module):
    """
    Graph Convolutional Network for predicting privacy risk scores.
    """
    def __init__(self, num_features, hidden_channels=64, dropout=0.5):
        super(GCNPrivacyPredictor, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third Graph Convolution Layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Final Linear Layer to produce privacy score
        x = self.lin(x)
        
        # Sigmoid activation to output scores between 0 and 1
        return torch.sigmoid(x).view(-1)

def load_facebook_data(ego_id, sensitive_attr_idx=None):
    """
    Load Facebook ego-network data for a specific user.
    
    Parameters:
    - ego_id: ID of the ego user
    - sensitive_attr_idx: Index of the feature to use as sensitive attribute
                          If None, a random feature will be selected
                          
    Returns:
    - data: PyTorch Geometric Data object with the graph
    - sensitive_attr_idx: Index of the sensitive attribute
    - feature_names: Names of all features
    """
    base_dir = "facebook"
    
    # Load edges
    edges_file = os.path.join(base_dir, f"{ego_id}.edges")
    edges_df = pd.read_csv(edges_file, sep=' ', header=None, names=['source', 'target'])
    
    # Load features
    feat_file = os.path.join(base_dir, f"{ego_id}.feat")
    feat_df = pd.read_csv(feat_file, sep=' ', header=None)
    
    # Load feature names
    featnames_file = os.path.join(base_dir, f"{ego_id}.featnames")
    feature_names = []
    with open(featnames_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(' ', 1)
            feature_names.append(name)
    
    # Convert node IDs and edges to proper format
    node_ids = feat_df[0].values
    features = feat_df.iloc[:, 1:].values
    
    # Create node ID mapping
    node_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    
    # Map edge IDs
    edge_index = []
    for _, row in edges_df.iterrows():
        if row['source'] in node_map and row['target'] in node_map:
            edge_index.append([node_map[row['source']], node_map[row['target']]])
            # Add the reverse edge for undirected graph
            edge_index.append([node_map[row['target']], node_map[row['source']]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Convert features to tensor
    x = torch.tensor(features, dtype=torch.float)
    
    # If no sensitive attribute specified, randomly select one
    if sensitive_attr_idx is None:
        # Choose an attribute with some variability (not all 0s or 1s)
        candidates = []
        for i in range(x.shape[1]):
            unique_vals = torch.unique(x[:, i])
            if len(unique_vals) > 1 and len(unique_vals) < x.shape[0] * 0.8:
                candidates.append(i)
        
        if candidates:
            sensitive_attr_idx = random.choice(candidates)
        else:
            sensitive_attr_idx = random.randint(0, x.shape[1] - 1)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data, sensitive_attr_idx, feature_names

def calculate_privacy_score(node_features, sensitive_attr_idx, neighbors_features):
    """
    Calculate the "true" privacy score for a node based on how easily its
    sensitive attribute can be inferred from its neighbors' features.
    
    A lower score means higher privacy risk.
    
    Parameters:
    - node_features: Features of the target node
    - sensitive_attr_idx: Index of the sensitive attribute
    - neighbors_features: Features of the node's neighbors
    
    Returns:
    - privacy_score: Value between 0 and 1
    """
    if len(neighbors_features) == 0:
        return 1.0  # No neighbors means no inference risk
    
    # Get the sensitive attribute value for the node
    sensitive_value = node_features[sensitive_attr_idx].item()
    
    # Check how many neighbors have the same sensitive value
    matching_neighbors = sum(1 for n_feat in neighbors_features 
                             if n_feat[sensitive_attr_idx].item() == sensitive_value)
    
    if len(neighbors_features) > 0:
        # The ratio of neighbors with different sensitive values indicates privacy
        privacy_score = 1.0 - (matching_neighbors / len(neighbors_features))
    else:
        privacy_score = 1.0  # No neighbors means no inference risk
    
    # Adjust privacy score to ensure it's between 0.1 and 0.9
    # (avoiding extreme values of 0 or 1 for numerical stability)
    privacy_score = 0.1 + 0.8 * privacy_score
    
    return privacy_score

def simulate_privacy_enhancement(data, privacy_scores, sensitive_attr_idx, privacy_threshold=0.3, 
                                intervention_rate=0.5):
    """
    Simulate privacy enhancement by generalizing the sensitive attribute
    for users with low privacy scores.
    
    Parameters:
    - data: PyTorch Geometric Data object
    - privacy_scores: Predicted privacy scores for each node
    - sensitive_attr_idx: Index of the sensitive attribute
    - privacy_threshold: Threshold below which to apply privacy enhancement
    - intervention_rate: Proportion of eligible nodes to enhance
    
    Returns:
    - enhanced_data: Data with enhanced privacy
    - new_privacy_scores: Recalculated privacy scores after enhancement
    - utility_loss: Measure of data utility lost due to enhancement
    """
    # Create a copy of the data to modify
    enhanced_features = data.x.clone()
    
    # Find nodes with privacy scores below threshold
    at_risk_nodes = (privacy_scores < privacy_threshold).nonzero(as_tuple=True)[0]
    
    # Randomly select a subset of these nodes to enhance
    num_to_enhance = int(len(at_risk_nodes) * intervention_rate)
    if num_to_enhance > 0:
        nodes_to_enhance = at_risk_nodes[torch.randperm(len(at_risk_nodes))[:num_to_enhance]]
        
        # Get the most common value for the sensitive attribute
        values, counts = torch.unique(data.x[:, sensitive_attr_idx], return_counts=True)
        most_common_value = values[counts.argmax()]
        
        # Apply generalization (set to most common value)
        enhanced_features[nodes_to_enhance, sensitive_attr_idx] = most_common_value
        
        # Calculate utility loss as proportion of modified nodes
        utility_loss = len(nodes_to_enhance) / len(data.x)
    else:
        utility_loss = 0.0
    
    # Create new data object with enhanced features
    enhanced_data = Data(x=enhanced_features, edge_index=data.edge_index)
    
    # Recalculate privacy scores based on enhanced data
    new_privacy_scores = []
    edge_index = enhanced_data.edge_index
    for node_idx in range(len(enhanced_data.x)):
        # Find neighbors of this node
        neighbors = []
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() == node_idx:
                neighbors.append(edge_index[1, i].item())
        
        # Get features of neighbors
        neighbors_features = [enhanced_data.x[n] for n in neighbors]
        
        # Calculate privacy score
        score = calculate_privacy_score(enhanced_data.x[node_idx], 
                                       sensitive_attr_idx, 
                                       neighbors_features)
        new_privacy_scores.append(score)
    
    new_privacy_scores = torch.tensor(new_privacy_scores, dtype=torch.float)
    
    return enhanced_data, new_privacy_scores, utility_loss

def train_model(model, data, privacy_scores, optimizer, epochs=100):
    """
    Train the GNN model to predict privacy scores.
    
    Parameters:
    - model: GCNPrivacyPredictor model
    - data: PyTorch Geometric Data object
    - privacy_scores: Ground truth privacy scores
    - optimizer: PyTorch optimizer
    - epochs: Number of training epochs
    
    Returns:
    - trained model
    """
    model.train()
    
    # Split nodes into train/validation sets
    n_nodes = data.x.shape[0]
    indices = list(range(n_nodes))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    
    # Best validation loss
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Calculate loss on training nodes
        loss = F.mse_loss(out[train_idx], privacy_scores[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.mse_loss(val_out[val_idx], privacy_scores[val_idx])
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        model.train()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, data, privacy_scores):
    """
    Evaluate the model's performance.
    
    Parameters:
    - model: Trained GCNPrivacyPredictor model
    - data: PyTorch Geometric Data object
    - privacy_scores: Ground truth privacy scores
    
    Returns:
    - mse: Mean squared error
    - predicted_scores: Model's predictions
    """
    model.eval()
    
    with torch.no_grad():
        predicted_scores = model(data.x, data.edge_index)
        mse = F.mse_loss(predicted_scores, privacy_scores).item()
        
        # Calculate binary metrics using a threshold of 0.5
        binary_true = (privacy_scores < 0.5).float()
        binary_pred = (predicted_scores < 0.5).float()
        
        accuracy = accuracy_score(binary_true, binary_pred)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(binary_true, 1 - predicted_scores)  # 1 - score since lower score = higher risk
        except:
            auc = 0.5  # Default if AUC can't be calculated
    
    return mse, accuracy, f1, auc, predicted_scores

def train_iterative_model(ego_id, sensitive_attr_idx=None, num_iterations=5, 
                        privacy_threshold=0.3, intervention_rate=0.5):
    """
    Train the GNN model iteratively, improving it through simulated privacy enhancements.
    
    Parameters:
    - ego_id: ID of the ego user in the Facebook dataset
    - sensitive_attr_idx: Index of the sensitive attribute (or None to select randomly)
    - num_iterations: Number of iteration cycles
    - privacy_threshold: Threshold below which to apply privacy enhancement
    - intervention_rate: Proportion of eligible nodes to enhance
    
    Returns:
    - final_model: The final trained GNN model
    - performance_history: Dictionary tracking performance metrics across iterations
    """
    # Load data
    data, sensitive_attr_idx, feature_names = load_facebook_data(ego_id, sensitive_attr_idx)
    
    print(f"Loaded network with {data.x.shape[0]} nodes, {data.edge_index.shape[1]//2} edges")
    print(f"Selected sensitive attribute: {feature_names[sensitive_attr_idx]}")
    
    # Calculate initial "ground truth" privacy scores
    initial_privacy_scores = []
    for node_idx in range(len(data.x)):
        # Find neighbors of this node
        neighbors = []
        for i in range(data.edge_index.shape[1]):
            if data.edge_index[0, i].item() == node_idx:
                neighbors.append(data.edge_index[1, i].item())
        
        # Get features of neighbors
        neighbors_features = [data.x[n] for n in neighbors]
        
        # Calculate privacy score
        score = calculate_privacy_score(data.x[node_idx], sensitive_attr_idx, neighbors_features)
        initial_privacy_scores.append(score)
    
    privacy_scores = torch.tensor(initial_privacy_scores, dtype=torch.float)
    
    # Initialize model
    model = GCNPrivacyPredictor(num_features=data.x.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Track performance across iterations
    performance_history = {
        'mse': [],
        'accuracy': [],
        'f1': [],
        'auc': [],
        'utility_loss': []
    }
    
    current_data = data
    current_privacy_scores = privacy_scores
    
    # Iterative training loop
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        
        # Train model on current data
        model = train_model(model, current_data, current_privacy_scores, optimizer)
        
        # Evaluate model
        mse, accuracy, f1, auc, predicted_scores = evaluate_model(model, current_data, current_privacy_scores)
        
        print(f"MSE: {mse:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Simulate privacy enhancement
        enhanced_data, new_privacy_scores, utility_loss = simulate_privacy_enhancement(
            current_data, predicted_scores, sensitive_attr_idx, 
            privacy_threshold, intervention_rate
        )
        
        print(f"Privacy enhancement applied. Utility loss: {utility_loss:.4f}")
        
        # Update current data for next iteration
        current_data = enhanced_data
        current_privacy_scores = new_privacy_scores
        
        # Record performance
        performance_history['mse'].append(mse)
        performance_history['accuracy'].append(accuracy)
        performance_history['f1'].append(f1)
        performance_history['auc'].append(auc)
        performance_history['utility_loss'].append(utility_loss)
    
    # Final evaluation
    final_mse, final_accuracy, final_f1, final_auc, _ = evaluate_model(model, current_data, current_privacy_scores)
    
    print("\nFinal Model Performance:")
    print(f"MSE: {final_mse:.4f}, Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}, AUC: {final_auc:.4f}")
    
    return model, performance_history, sensitive_attr_idx, feature_names

if __name__ == "__main__":
    # Select an ego network to use
    ego_id = "0"  # You can change this to any ID in the facebook directory
    
    # Train the model iteratively
    model, performance_history, sensitive_attr_idx, feature_names = train_iterative_model(
        ego_id=ego_id,
        num_iterations=5,
        privacy_threshold=0.3,
        intervention_rate=0.6
    )
    
    print(f"\nUsed sensitive attribute: {feature_names[sensitive_attr_idx]}")
    
    # Print performance history
    print("\nPerformance over iterations:")
    for i, (mse, acc, f1, auc, util_loss) in enumerate(zip(
        performance_history['mse'],
        performance_history['accuracy'],
        performance_history['f1'],
        performance_history['auc'],
        performance_history['utility_loss']
    )):
        print(f"Iteration {i+1}: MSE={mse:.4f}, Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Utility Loss={util_loss:.4f}") 