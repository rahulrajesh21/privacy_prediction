import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loader import FacebookDataLoader
from twitter_loader import TwitterDataLoader
from multiview_model import MultiViewPrivacyGNN
from model import PrivacyGNNWithExplanation

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-View Privacy Risk Assessment GNN')
    parser.add_argument('--fb_data_dir', type=str, default='facebook', help='Path to Facebook dataset')
    parser.add_argument('--tw_data_dir', type=str, default='twitter', help='Path to Twitter dataset')
    parser.add_argument('--fb_ego_id', type=str, default=None, help='Specific Facebook ego network ID to train on')
    parser.add_argument('--tw_ego_id', type=str, default=None, help='Specific Twitter ego network ID to train on')
    parser.add_argument('--tw_sample_size', type=int, default=5, help='Number of Twitter networks to sample')
    parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'], 
                        help='Type of GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='multiview_output', help='Output directory')
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

def train_epoch(model, fb_data, tw_data, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    
    # Move data to device
    fb_data = fb_data.to(device)
    tw_data = tw_data.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    pred = model(fb_data=fb_data, tw_data=tw_data)
    
    # Create target by using only Facebook privacy risk (since prediction is based on Facebook nodes)
    # We can't average with Twitter because the sizes might be different
    target = fb_data.privacy_risk
    
    # Calculate loss
    loss = model.loss(pred.squeeze(), target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, fb_data, tw_data, device):
    """Evaluate the model."""
    model.eval()
    
    # Move data to device
    fb_data = fb_data.to(device)
    tw_data = tw_data.to(device)
    
    # Forward pass
    with torch.no_grad():
        pred = model(fb_data=fb_data, tw_data=tw_data)
        
        # Use only Facebook privacy risk as target
        target = fb_data.privacy_risk
        
        mse = ((pred.squeeze() - target) ** 2).mean().item()
    
    return mse

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Facebook data
    print("Loading Facebook dataset...")
    fb_loader = FacebookDataLoader(data_dir=args.fb_data_dir)
    
    # If specific ego ID is provided, load only that network
    if args.fb_ego_id:
        fb_networks = [fb_loader.load_ego_network(args.fb_ego_id)]
        print(f"Loaded Facebook ego network {args.fb_ego_id}")
    else:
        fb_networks = fb_loader.load_all_networks()
        print(f"Loaded {len(fb_networks)} Facebook ego networks")
    
    # Load Twitter data
    print("Loading Twitter dataset...")
    tw_loader = TwitterDataLoader(data_dir=args.tw_data_dir, sample_size=args.tw_sample_size)
    
    # If specific ego ID is provided, load only that network
    if args.tw_ego_id:
        tw_networks = [tw_loader.load_ego_network(args.tw_ego_id)]
        print(f"Loaded Twitter ego network {args.tw_ego_id}")
    else:
        tw_networks = tw_loader.load_all_networks()
        print(f"Loaded {len(tw_networks)} Twitter ego networks")
    
    # Define sensitive attributes
    sensitive_keywords = [
        'gender', 'birth', 'age', 'location', 'hometown', 'address',
        'education', 'school', 'university', 'work', 'employer',
        'religion', 'political', 'relationship', 'interested_in',
        'family', 'languages', 'email', 'phone'
    ]
    
    # Process Facebook networks
    processed_fb_networks = []
    for network in fb_networks:
        # Get feature names
        feature_names = network.feature_names
        
        # Identify sensitive attribute indices
        sensitive_indices = identify_sensitive_attributes(feature_names, sensitive_keywords)
        
        # Create privacy risk labels
        network = fb_loader.create_privacy_labels([network], sensitive_indices)[0]
        processed_fb_networks.append(network)
    
    # Process Twitter networks
    processed_tw_networks = []
    for network in tw_networks:
        # Get feature names
        feature_names = network.feature_names
        
        # Identify sensitive attribute indices
        sensitive_indices = identify_sensitive_attributes(feature_names, sensitive_keywords)
        
        # Create privacy risk labels
        network = tw_loader.create_privacy_labels([network], sensitive_indices)[0]
        processed_tw_networks.append(network)
    
    print(f"Identified sensitive attributes and created privacy risk labels")
    
    # Split data into train, validation, and test sets
    # Handle the case when there's only one network
    if len(processed_fb_networks) == 1:
        fb_train = fb_val = fb_test = processed_fb_networks
    else:
        fb_train, fb_test = train_test_split(processed_fb_networks, test_size=0.2, random_state=args.seed)
        fb_train, fb_val = train_test_split(fb_train, test_size=0.25, random_state=args.seed)
    
    if len(processed_tw_networks) == 1:
        tw_train = tw_val = tw_test = processed_tw_networks
    else:
        tw_train, tw_test = train_test_split(processed_tw_networks, test_size=0.2, random_state=args.seed)
        tw_train, tw_val = train_test_split(tw_train, test_size=0.25, random_state=args.seed)
    
    print(f"Split data into train, validation, and test sets")
    
    # Train models for each Facebook-Twitter network pair
    for fb_idx, fb_network in enumerate(fb_train):
        for tw_idx, tw_network in enumerate(tw_train):
            print(f"\nTraining model for Facebook network {fb_network.ego_id} and Twitter network {tw_network.ego_id}")
            
            # Determine input dimensions
            fb_input_dim = fb_network.x.size(1)
            tw_input_dim = tw_network.x.size(1)
            
            # Create multi-view model
            model = MultiViewPrivacyGNN(
                fb_input_dim=fb_input_dim,
                tw_input_dim=tw_input_dim,
                fb_feature_names=fb_network.feature_names,
                tw_feature_names=tw_network.feature_names,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                conv_type=args.conv_type
            )
            
            model = model.to(device)
            print(f"Created multi-view model with {args.conv_type} layers")
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            
            # Training loop
            best_val_mse = float('inf')
            train_losses = []
            val_mses = []
            
            print("Starting training...")
            for epoch in range(args.epochs):
                # Train
                train_loss = train_epoch(model, fb_network, tw_network, optimizer, device)
                train_losses.append(train_loss)
                
                # Evaluate on validation set
                val_mse = evaluate(model, fb_val[0], tw_val[0], device)
                val_mses.append(val_mse)
                
                # Save best model
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_fb{fb_network.ego_id}_tw{tw_network.ego_id}.pt'))
                
                print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}")
            
            # Save final model
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'final_model_fb{fb_network.ego_id}_tw{tw_network.ego_id}.pt'))
            
            # Plot training curves
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title(f'Training Loss (FB:{fb_network.ego_id}, TW:{tw_network.ego_id})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(val_mses)
            plt.title(f'Validation MSE (FB:{fb_network.ego_id}, TW:{tw_network.ego_id})')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f'training_curves_fb{fb_network.ego_id}_tw{tw_network.ego_id}.png'))
            plt.close()
            
            # Generate explanations
            print("Analyzing feature importance...")
            model.eval()
            
            # Move networks to device
            fb_network = fb_network.to(device)
            tw_network = tw_network.to(device)
            
            # Get explanation
            explanation = model.explain_prediction(fb_network, tw_network)
            
            # Print top important features from both platforms
            print("\nTop important features for privacy risk:")
            for i, feature in enumerate(explanation["combined_important_features"][:5]):
                print(f"{i+1}. [{feature['platform']}] {feature['feature']} (Importance: {feature['importance']:.4f})")
            
            # Save feature importance data
            with open(os.path.join(args.output_dir, f'feature_importance_fb{fb_network.ego_id}_tw{tw_network.ego_id}.txt'), 'w') as f:
                f.write("Multi-view Privacy Risk Assessment Feature Importance\n")
                f.write("="*50 + "\n\n")
                
                f.write("Top Facebook Features:\n")
                for i, feature in enumerate(explanation["facebook"]["important_features"]):
                    f.write(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})\n")
                
                f.write("\nTop Twitter Features:\n")
                for i, feature in enumerate(explanation["twitter"]["important_features"]):
                    f.write(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})\n")
                
                f.write("\nCombined Important Features:\n")
                for i, feature in enumerate(explanation["combined_important_features"]):
                    f.write(f"{i+1}. [{feature['platform']}] {feature['feature']} (Importance: {feature['importance']:.4f})\n")
    
    print("\nTraining complete. Models saved to output directory.")

if __name__ == '__main__':
    main() 