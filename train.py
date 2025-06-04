import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loader import FacebookDataLoader
from model import PrivacyGNN, PrivacyGNNWithExplanation

def parse_args():
    parser = argparse.ArgumentParser(description='Train Privacy Risk Assessment GNN')
    parser.add_argument('--data_dir', type=str, default='facebook', help='Path to Facebook dataset')
    parser.add_argument('--model_type', type=str, default='explainable', choices=['basic', 'explainable'], 
                        help='Type of model to train')
    parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'], 
                        help='Type of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--ego_id', type=str, default=None, help='Specific ego network ID to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def identify_sensitive_attributes(feature_names, sensitive_keywords):
    """Identify sensitive attributes based on keywords."""
    sensitive_indices = []
    for i, name in enumerate(feature_names):
        # Convert name to string if it's not already
        name_str = str(name) if name is not None else ""
        for keyword in sensitive_keywords:
            if keyword.lower() in name_str.lower():
                sensitive_indices.append(i)
                break
    return sensitive_indices

def train_epoch(model, data, optimizer, device):
    """Train the model for one epoch on a single network."""
    model.train()
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    
    # Calculate loss
    loss = model.loss(pred.squeeze(), data.privacy_risk)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_network(model, data, device):
    """Evaluate the model on a single network."""
    model.eval()
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        mse = ((pred.squeeze() - data.privacy_risk) ** 2).mean().item()
    
    return mse

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Facebook dataset...")
    loader = FacebookDataLoader(data_dir=args.data_dir)
    
    # If specific ego ID is provided, load only that network
    if args.ego_id:
        networks = [loader.load_ego_network(args.ego_id)]
        print(f"Loaded ego network {args.ego_id}")
    else:
        networks = loader.load_all_networks()
        print(f"Loaded {len(networks)} ego networks")
    
    # Define sensitive attributes
    sensitive_keywords = [
        'gender', 'birth', 'age', 'location', 'hometown', 'address',
        'education', 'school', 'university', 'work', 'employer',
        'religion', 'political', 'relationship', 'interested_in',
        'family', 'languages', 'email', 'phone'
    ]
    
    # Process each network individually
    processed_networks = []
    for network in networks:
        # Get feature names
        feature_names = network.feature_names
        
        # Identify sensitive attribute indices
        sensitive_indices = identify_sensitive_attributes(feature_names, sensitive_keywords)
        
        # Create privacy risk labels
        network = loader.create_privacy_labels([network], sensitive_indices)[0]
        processed_networks.append(network)
    
    print(f"Identified sensitive attributes and created privacy risk labels")
    
    # Split data into train, validation, and test sets if we have multiple networks
    if len(processed_networks) > 1:
        train_networks, test_networks = train_test_split(processed_networks, test_size=0.2, random_state=args.seed)
        train_networks, val_networks = train_test_split(train_networks, test_size=0.25, random_state=args.seed)
        print(f"Split data into {len(train_networks)} train, {len(val_networks)} validation, and {len(test_networks)} test networks")
    else:
        # If we only have one network, use it for both training and validation
        train_networks = processed_networks
        val_networks = processed_networks
        test_networks = processed_networks
        print("Using the single network for training, validation, and testing")
    
    # Train a separate model for each network
    for network_idx, network in enumerate(train_networks):
        print(f"\nTraining model for network {network_idx+1}/{len(train_networks)} (Ego ID: {network.ego_id})")
        
        # Determine input dimension for this network
        input_dim = network.x.size(1)
        
        # Create model
        if args.model_type == 'basic':
            model = PrivacyGNN(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                conv_type=args.conv_type
            )
        else:  # explainable
            model = PrivacyGNNWithExplanation(
                input_dim=input_dim,
                feature_names=network.feature_names,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                conv_type=args.conv_type
            )
        
        model = model.to(device)
        print(f"Created {args.model_type} model with {args.conv_type} layers")
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Training loop
        best_val_mse = float('inf')
        train_losses = []
        val_mses = []
        
        print("Starting training...")
        for epoch in range(args.epochs):
            # Train on this network
            train_loss = train_epoch(model, network, optimizer, device)
            train_losses.append(train_loss)
            
            # Evaluate on validation networks with matching feature dimensions
            val_mses_epoch = []
            for val_network in val_networks:
                if val_network.x.size(1) == input_dim:
                    val_mse = evaluate_network(model, val_network, device)
                    val_mses_epoch.append(val_mse)
            
            # If no validation networks with matching dimensions, use training loss
            if val_mses_epoch:
                val_mse = np.mean(val_mses_epoch)
            else:
                val_mse = train_loss
            
            val_mses.append(val_mse)
            
            # Save best model
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_{network.ego_id}.pt'))
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}")
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'final_model_{network.ego_id}.pt'))
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'Training Loss (Network {network.ego_id})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_mses)
        plt.title(f'Validation MSE (Network {network.ego_id})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'training_curves_{network.ego_id}.png'))
        plt.close()
        
        # If using explainable model, analyze feature importance
        if args.model_type == 'explainable':
            print("Analyzing feature importance...")
            model.eval()
            
            # Get explanation
            network = network.to(device)
            explanation = model.explain_prediction(network.x, network.edge_index)
            
            # Print top important features
            print("\nTop 5 most important features for privacy risk:")
            for i, feature in enumerate(explanation['important_features']):
                print(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})")
            
            # Save feature importance plot
            plt.figure(figsize=(12, 6))
            importance = explanation['all_feature_weights']
            indices = np.argsort(importance)[-20:]  # Top 20 features
            
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [network.feature_names[i] for i in indices])
            plt.title(f'Top 20 Feature Importance for Privacy Risk (Network {network.ego_id})')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'feature_importance_{network.ego_id}.png'))
            plt.close()
    
    print("\nTraining complete. Models saved to output directory.")

if __name__ == '__main__':
    main() 