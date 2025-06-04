import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from gnn_privacy_predictor import load_facebook_data, train_iterative_model

def plot_performance_metrics(performance_history):
    """
    Plot the evolution of model performance metrics over iterations.
    
    Parameters:
    - performance_history: Dictionary with lists of metrics per iteration
    """
    plt.figure(figsize=(12, 8))
    
    # Plot MSE
    plt.subplot(2, 2, 1)
    plt.plot(performance_history['mse'], 'o-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE over Iterations')
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(performance_history['accuracy'], 'o-', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations')
    plt.grid(True)
    
    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(performance_history['f1'], 'o-', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Iterations')
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 4)
    plt.plot(performance_history['auc'], 'o-', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('AUC over Iterations')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

def plot_privacy_utility_tradeoff(performance_history):
    """
    Plot the privacy-utility tradeoff.
    
    Parameters:
    - performance_history: Dictionary with metrics per iteration
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate cumulative utility loss
    cumulative_utility_loss = np.cumsum(performance_history['utility_loss'])
    
    # Plot AUC (privacy protection metric) vs cumulative utility loss
    plt.plot(cumulative_utility_loss, performance_history['auc'], 'o-', color='blue', linewidth=2)
    plt.xlabel('Cumulative Utility Loss')
    plt.ylabel('AUC (Privacy Protection)')
    plt.title('Privacy-Utility Tradeoff')
    plt.grid(True)
    
    # Add iteration labels
    for i, (x, y) in enumerate(zip(cumulative_utility_loss, performance_history['auc'])):
        plt.annotate(f'Iter {i+1}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig('privacy_utility_tradeoff.png')
    plt.close()

def visualize_network(data, privacy_scores, ego_id, sensitive_attr_idx):
    """
    Visualize the network with nodes colored by their privacy scores.
    
    Parameters:
    - data: PyTorch Geometric Data object
    - privacy_scores: Privacy scores for each node
    - ego_id: ID of the ego user
    - sensitive_attr_idx: Index of the sensitive attribute
    """
    # Convert to networkx for visualization
    G = to_networkx(data, to_undirected=True)
    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Convert privacy scores to colors (red for high risk, green for low risk)
    colors = []
    for score in privacy_scores:
        # Red (low score/high risk) to Green (high score/low risk)
        r = max(0, min(1, 2 * (0.5 - score.item())))
        g = max(0, min(1, 2 * (score.item() - 0.5)))
        colors.append((r, g, 0))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, alpha=0.8)
    
    # Draw edges with low opacity for better visibility
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    # Add a color bar to show privacy score scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='vertical')
    cbar.set_label('Privacy Score (0=High Risk, 1=Low Risk)')
    
    plt.title(f'Network Privacy Visualization (Ego ID: {ego_id})\nSensitive Attribute: {sensitive_attr_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'network_privacy_ego_{ego_id}.png')
    plt.close()

def plot_sensitive_attribute_distribution(data, sensitive_attr_idx, feature_names):
    """
    Plot the distribution of the sensitive attribute values.
    
    Parameters:
    - data: PyTorch Geometric Data object
    - sensitive_attr_idx: Index of the sensitive attribute
    - feature_names: Names of the features
    """
    # Get sensitive attribute values
    sensitive_values = data.x[:, sensitive_attr_idx].numpy()
    
    # Get unique values and their counts
    unique_values, counts = np.unique(sensitive_values, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Attribute Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of Sensitive Attribute: {feature_names[sensitive_attr_idx]}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sensitive_attribute_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Select an ego network
    ego_id = "0"  # You can change this to any ID in the facebook directory
    
    # Option 1: Run the full iterative training and visualization
    model, performance_history, sensitive_attr_idx, feature_names = train_iterative_model(
        ego_id=ego_id,
        num_iterations=5,
        privacy_threshold=0.3,
        intervention_rate=0.6
    )
    
    # Plot performance metrics
    plot_performance_metrics(performance_history)
    
    # Plot privacy-utility tradeoff
    plot_privacy_utility_tradeoff(performance_history)
    
    # Load data for visualization
    data, _, _ = load_facebook_data(ego_id, sensitive_attr_idx)
    
    # Run model on data
    model.eval()
    with torch.no_grad():
        predicted_scores = model(data.x, data.edge_index)
    
    # Visualize network with privacy scores
    visualize_network(data, predicted_scores, ego_id, feature_names[sensitive_attr_idx])
    
    # Plot sensitive attribute distribution
    plot_sensitive_attribute_distribution(data, sensitive_attr_idx, feature_names)
    
    print("Visualization complete. Check the current directory for output images.") 