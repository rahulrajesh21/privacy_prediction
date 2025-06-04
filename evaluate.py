import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from data_loader import FacebookDataLoader
from model import PrivacyGNN, PrivacyGNNWithExplanation

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Privacy Risk Assessment GNN')
    parser.add_argument('--data_dir', type=str, default='facebook', help='Path to Facebook dataset')
    parser.add_argument('--model_path', type=str, default='output/best_model.pt', help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='explainable', choices=['basic', 'explainable'], 
                        help='Type of model to use')
    parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'], 
                        help='Type of GNN layer used in training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--ego_id', type=str, default=None, help='Specific ego network ID to evaluate')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

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

def visualize_network_risk(data, risk_scores, node_mapping, ego_id, output_path):
    """Visualize network with privacy risk color coding."""
    # Create NetworkX graph
    G = nx.Graph()
    
    # Reverse node mapping (index -> original node ID)
    rev_mapping = {idx: node_id for node_id, idx in node_mapping.items()}
    
    # Add nodes with risk scores
    for idx, risk in enumerate(risk_scores):
        # Normalized risk to a color (red = high risk, green = low risk)
        color = plt.cm.RdYlGn_r(float(risk))
        G.add_node(rev_mapping[idx], risk=float(risk), color=color)
    
    # Add edges
    edge_index = data.edge_index.cpu().numpy()
    edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if (src, dst) not in edges and (dst, src) not in edges:
            edges.add((src, dst))
            G.add_edge(rev_mapping[src], rev_mapping[dst])
    
    # Create plot
    plt.figure(figsize=(12, 12))
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with color based on risk score
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # Draw ego node larger
    node_sizes = [300 if n != int(ego_id) else 600 for n in G.nodes()]
    
    # Draw network
    nx.draw_networkx(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes,
        with_labels=False, 
        edge_color='gray', 
        alpha=0.8
    )
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Privacy Risk Score')
    
    plt.title(f'Privacy Risk Assessment for Ego Network {ego_id}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_network_privacy(model, data, feature_names, ego_id, output_dir):
    """Analyze privacy risks in a network and generate reports."""
    device = data.x.device
    model.eval()
    
    # Get privacy risk predictions
    with torch.no_grad():
        if hasattr(model, 'explain_prediction'):
            risk_scores, feature_weights = model.forward(data.x, data.edge_index, return_attention=True)
            explanation = model.explain_prediction(data.x, data.edge_index)
        else:
            risk_scores = model(data.x, data.edge_index)
            feature_weights = None
            explanation = None
    
    risk_scores = risk_scores.squeeze().cpu().numpy()
    
    # Compute network-level statistics
    avg_risk = np.mean(risk_scores)
    max_risk = np.max(risk_scores)
    min_risk = np.min(risk_scores)
    
    # Sort nodes by risk score
    node_indices = np.argsort(risk_scores)[::-1]  # Descending order
    
    # Reverse node mapping (index -> original node ID)
    rev_mapping = {idx: node_id for node_id, idx in data.node_mapping.items()}
    
    # Generate privacy risk visualization
    vis_path = os.path.join(output_dir, f'privacy_risk_network_{ego_id}.png')
    visualize_network_risk(data, risk_scores, data.node_mapping, ego_id, vis_path)
    
    # Generate privacy risk report
    report_path = os.path.join(output_dir, f'privacy_risk_report_{ego_id}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Privacy Risk Assessment Report for Network {ego_id}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Network-level Statistics:\n")
        f.write(f"Average Privacy Risk: {avg_risk:.4f}\n")
        f.write(f"Maximum Privacy Risk: {max_risk:.4f}\n")
        f.write(f"Minimum Privacy Risk: {min_risk:.4f}\n\n")
        
        f.write("Top 10 Most At-Risk Nodes:\n")
        for i, idx in enumerate(node_indices[:10]):
            original_id = rev_mapping[idx]
            f.write(f"{i+1}. Node {original_id}: Risk Score {risk_scores[idx]:.4f}\n")
        f.write("\n")
        
        # Include feature importance information if available
        if explanation:
            f.write("Feature Importance Analysis:\n")
            f.write("Top 5 most privacy-sensitive features:\n")
            for i, feature in enumerate(explanation['important_features']):
                f.write(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})\n")
            f.write("\n")
            
            f.write("Privacy Recommendations:\n")
            f.write("Based on the feature importance analysis, consider limiting the exposure of the following information:\n")
            for feature in explanation['important_features'][:3]:
                f.write(f"- {feature['feature']}\n")
            f.write("\n")
        
        f.write("Node-specific Privacy Recommendations:\n")
        for i, idx in enumerate(node_indices[:5]):
            original_id = rev_mapping[idx]
            f.write(f"Node {original_id} (Risk: {risk_scores[idx]:.4f}):\n")
            f.write("  - Consider reviewing privacy settings and limiting network visibility\n")
            f.write("  - Reduce exposure of sensitive personal information\n")
            f.write("  - Review friend connections and consider creating separate circles/groups\n\n")
    
    print(f"Privacy risk analysis for network {ego_id} completed.")
    print(f"Report saved to {report_path}")
    print(f"Visualization saved to {vis_path}")
    
    return {
        'risk_scores': risk_scores,
        'avg_risk': avg_risk,
        'max_risk': max_risk,
        'min_risk': min_risk,
        'explanation': explanation
    }

def main():
    args = parse_args()
    
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
    
    # Get feature names from first network
    feature_names = networks[0].feature_names if networks else []
    
    # Identify sensitive attribute indices
    sensitive_indices = identify_sensitive_attributes(feature_names, sensitive_keywords)
    print(f"Identified {len(sensitive_indices)} sensitive attributes")
    
    # Create privacy risk labels (for reference, not used in evaluation)
    networks = loader.create_privacy_labels(networks, sensitive_indices)
    
    # Determine input dimension
    input_dim = networks[0].x.size(1) if networks else 0
    
    # Create model with the same architecture as used in training
    if args.model_type == 'basic':
        model = PrivacyGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            conv_type=args.conv_type
        )
    else:  # explainable
        model = PrivacyGNNWithExplanation(
            input_dim=input_dim,
            feature_names=feature_names,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            conv_type=args.conv_type
        )
    
    # Load trained model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.model_type} model from {args.model_path}")
    
    # Analyze each network
    all_avg_risks = []
    all_max_risks = []
    
    for i, data in enumerate(tqdm(networks, desc="Analyzing networks")):
        data = data.to(device)
        ego_id = str(data.ego_id)
        
        results = analyze_network_privacy(model, data, feature_names, ego_id, args.output_dir)
        all_avg_risks.append(results['avg_risk'])
        all_max_risks.append(results['max_risk'])
    
    # Create summary visualization
    if len(networks) > 1:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_avg_risks, bins=10, alpha=0.7, color='blue')
        plt.axvline(np.mean(all_avg_risks), color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Average Privacy Risk')
        plt.xlabel('Average Risk Score')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(all_max_risks, bins=10, alpha=0.7, color='orange')
        plt.axvline(np.mean(all_max_risks), color='red', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Maximum Privacy Risk')
        plt.xlabel('Maximum Risk Score')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'privacy_risk_summary.png'))
        
        # Generate overall summary report
        with open(os.path.join(args.output_dir, 'privacy_risk_summary.txt'), 'w') as f:
            f.write("Privacy Risk Assessment Summary\n")
            f.write("="*40 + "\n\n")
            
            f.write(f"Number of networks analyzed: {len(networks)}\n")
            f.write(f"Overall average privacy risk: {np.mean(all_avg_risks):.4f}\n")
            f.write(f"Overall maximum privacy risk: {np.max(all_max_risks):.4f}\n\n")
            
            f.write("Networks with highest average privacy risk:\n")
            high_risk_indices = np.argsort(all_avg_risks)[-5:][::-1]
            for i, idx in enumerate(high_risk_indices):
                network = networks[idx]
                f.write(f"{i+1}. Network {network.ego_id}: {all_avg_risks[idx]:.4f}\n")
            
            f.write("\nPrivacy Recommendations:\n")
            f.write("1. Review and limit the visibility of sensitive personal information\n")
            f.write("2. Be cautious about connecting with unknown users\n")
            f.write("3. Regularly audit your privacy settings\n")
            f.write("4. Consider separating personal and professional networks\n")
            f.write("5. Limit the amount of personal information shared on public profiles\n")

if __name__ == '__main__':
    main() 