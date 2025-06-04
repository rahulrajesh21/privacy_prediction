#!/usr/bin/env python3
import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from data_loader import FacebookDataLoader
from model import PrivacyGNNWithExplanation
from evaluate import analyze_network_privacy, identify_sensitive_attributes

def parse_args():
    parser = argparse.ArgumentParser(description='Quick Privacy Risk Assessment')
    parser.add_argument('--ego_id', type=str, required=True, help='Ego network ID to assess')
    parser.add_argument('--data_dir', type=str, default='facebook', help='Path to Facebook dataset')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model (default: output/best_model_{ego_id}.pt)')
    parser.add_argument('--output_dir', type=str, default='privacy_reports', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading ego network {args.ego_id}...")
    loader = FacebookDataLoader(data_dir=args.data_dir)
    network = loader.load_ego_network(args.ego_id)
    
    # Define sensitive attributes
    sensitive_keywords = [
        'gender', 'birth', 'age', 'location', 'hometown', 'address',
        'education', 'school', 'university', 'work', 'employer',
        'religion', 'political', 'relationship', 'interested_in',
        'family', 'languages', 'email', 'phone'
    ]
    
    # Get feature names
    feature_names = network.feature_names
    
    # Identify sensitive attribute indices
    sensitive_indices = identify_sensitive_attributes(feature_names, sensitive_keywords)
    print(f"Identified {len(sensitive_indices)} sensitive attributes")
    
    # Create privacy risk labels
    network = loader.create_privacy_labels([network], sensitive_indices)[0]
    
    # Determine input dimension
    input_dim = network.x.size(1)
    
    # Create explainable model
    model = PrivacyGNNWithExplanation(
        input_dim=input_dim,
        feature_names=feature_names,
        hidden_dim=64,
        num_layers=3,
        conv_type='GAT'
    )
    
    # Determine model path
    if args.model_path is None:
        model_path = os.path.join('output', f'best_model_{args.ego_id}.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join('output', f'final_model_{args.ego_id}.pt')
    else:
        model_path = args.model_path
    
    # Load trained model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"No trained model found at {model_path}. Using untrained model for demonstration.")
    
    model = model.to(device)
    model.eval()
    
    # Move network to device
    network = network.to(device)
    
    # Analyze privacy risk
    print(f"Analyzing privacy risk for network {args.ego_id}...")
    results = analyze_network_privacy(model, network, feature_names, args.ego_id, args.output_dir)
    
    # Print summary results to console
    print("\n" + "="*50)
    print(f"PRIVACY RISK ASSESSMENT SUMMARY FOR NETWORK {args.ego_id}")
    print("="*50)
    
    print(f"\nNetwork-level statistics:")
    print(f"  Average Privacy Risk: {results['avg_risk']:.4f}")
    print(f"  Maximum Privacy Risk: {results['max_risk']:.4f}")
    print(f"  Minimum Privacy Risk: {results['min_risk']:.4f}")
    
    if results['explanation']:
        print("\nTop privacy-sensitive features:")
        table_data = []
        for i, feature in enumerate(results['explanation']['important_features']):
            table_data.append([i+1, feature['feature'], f"{feature['importance']:.4f}"])
        
        print(tabulate(table_data, headers=["Rank", "Feature", "Importance"], tablefmt="grid"))
    
    # Show top 10 highest risk nodes
    risk_scores = results['risk_scores']
    node_indices = np.argsort(risk_scores)[::-1]  # Descending order
    
    # Reverse node mapping (index -> original node ID)
    rev_mapping = {idx: node_id for node_id, idx in network.node_mapping.items()}
    
    print("\nTop 5 highest risk users:")
    table_data = []
    for i, idx in enumerate(node_indices[:5]):
        original_id = rev_mapping[idx]
        table_data.append([i+1, original_id, f"{risk_scores[idx]:.4f}"])
    
    print(tabulate(table_data, headers=["Rank", "User ID", "Risk Score"], tablefmt="grid"))
    
    print("\nPrivacy recommendations:")
    print("1. Limit sharing of sensitive personal information")
    print("2. Review connection privacy settings")
    print("3. Consider compartmentalizing social circles")
    
    print(f"\nDetailed report saved to: {args.output_dir}/privacy_risk_report_{args.ego_id}.txt")
    print(f"Network visualization saved to: {args.output_dir}/privacy_risk_network_{args.ego_id}.png")

if __name__ == '__main__':
    main() 