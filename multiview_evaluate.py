import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from data_loader import FacebookDataLoader
from twitter_loader import TwitterDataLoader
from multiview_model import MultiViewPrivacyGNN
from evaluate import identify_sensitive_attributes

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multi-View Privacy Risk Assessment GNN')
    parser.add_argument('--fb_ego_id', type=str, required=True, help='Facebook ego network ID to assess')
    parser.add_argument('--tw_ego_id', type=str, required=True, help='Twitter ego network ID to assess')
    parser.add_argument('--fb_data_dir', type=str, default='facebook', help='Path to Facebook dataset')
    parser.add_argument('--tw_data_dir', type=str, default='twitter', help='Path to Twitter dataset')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to trained model (default: multiview_output/best_model_fb{fb_ego_id}_tw{tw_ego_id}.pt)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT', 'SAGE'], 
                        help='Type of GNN layer')
    parser.add_argument('--output_dir', type=str, default='multiview_reports', help='Output directory')
    return parser.parse_args()

def visualize_privacy_comparison(fb_risks, tw_risks, combined_risks, fb_ego_id, tw_ego_id, output_path):
    """Visualize privacy risk comparison between Facebook, Twitter, and combined model."""
    plt.figure(figsize=(12, 8))
    
    # Create histogram of privacy risks
    plt.subplot(2, 1, 1)
    plt.hist(fb_risks, alpha=0.5, bins=20, label='Facebook', color='blue')
    
    # Only include Twitter risks if they're the same size as Facebook risks
    if len(tw_risks) >= len(fb_risks):
        plt.hist(tw_risks[:len(fb_risks)], alpha=0.5, bins=20, label='Twitter', color='green')
    
    plt.hist(combined_risks, alpha=0.5, bins=20, label='Combined', color='red')
    plt.xlabel('Privacy Risk Score')
    plt.ylabel('Count')
    plt.title(f'Privacy Risk Distribution Comparison (FB:{fb_ego_id}, TW:{tw_ego_id})')
    plt.legend()
    
    # Create scatter plot of Facebook vs Twitter vs Combined risks
    plt.subplot(2, 1, 2)
    plt.scatter(fb_risks, combined_risks, alpha=0.5, label='Facebook vs Combined', color='purple')
    
    # Only include Twitter risks if they're the same size as Facebook risks
    if len(tw_risks) >= len(fb_risks):
        plt.scatter(tw_risks[:len(fb_risks)], combined_risks, alpha=0.5, label='Twitter vs Combined', color='orange')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('Platform Risk Score')
    plt.ylabel('Combined Risk Score')
    plt.title('Platform vs Combined Privacy Risk Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_multiview_privacy(model, fb_data, tw_data, fb_ego_id, tw_ego_id, output_dir):
    """Analyze privacy risks using the multi-view model."""
    device = fb_data.x.device
    model.eval()
    
    # Get privacy risk predictions
    with torch.no_grad():
        # Get combined risk predictions
        combined_risks = model(fb_data=fb_data, tw_data=tw_data)
        
        # Get individual platform predictions
        fb_risks = model(x=fb_data.x, edge_index=fb_data.edge_index, platform="facebook")
        tw_risks = model(x=tw_data.x, edge_index=tw_data.edge_index, platform="twitter")
        
        # Get explanation
        explanation = model.explain_prediction(fb_data, tw_data)
    
    # Convert to numpy arrays
    combined_risks = combined_risks.squeeze().cpu().numpy()
    fb_risks = fb_risks.squeeze().cpu().numpy()
    tw_risks = tw_risks.squeeze().cpu().numpy()
    
    # Compute statistics
    avg_combined = np.mean(combined_risks)
    max_combined = np.max(combined_risks)
    min_combined = np.min(combined_risks)
    
    avg_fb = np.mean(fb_risks)
    max_fb = np.max(fb_risks)
    min_fb = np.min(fb_risks)
    
    # For Twitter, only compute stats if there are enough nodes
    if len(tw_risks) >= len(fb_risks):
        avg_tw = np.mean(tw_risks[:len(fb_risks)])
        max_tw = np.max(tw_risks[:len(fb_risks)])
        min_tw = np.min(tw_risks[:len(fb_risks)])
    else:
        avg_tw = np.mean(tw_risks)
        max_tw = np.max(tw_risks)
        min_tw = np.min(tw_risks)
    
    # Visualize privacy risk comparison
    vis_path = os.path.join(output_dir, f'privacy_comparison_fb{fb_ego_id}_tw{tw_ego_id}.png')
    visualize_privacy_comparison(fb_risks, tw_risks, combined_risks, fb_ego_id, tw_ego_id, vis_path)
    
    # Sort nodes by combined risk score
    node_indices = np.argsort(combined_risks)[::-1]  # Descending order
    
    # Reverse node mapping (index -> original node ID)
    fb_rev_mapping = {idx: node_id for node_id, idx in fb_data.node_mapping.items()}
    
    # Generate privacy risk report
    report_path = os.path.join(output_dir, f'multiview_privacy_report_fb{fb_ego_id}_tw{tw_ego_id}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Multi-view Privacy Risk Assessment Report\n")
        f.write(f"Facebook Network: {fb_ego_id}, Twitter Network: {tw_ego_id}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Network-level Statistics:\n")
        f.write(f"Combined Model:\n")
        f.write(f"  Average Privacy Risk: {avg_combined:.4f}\n")
        f.write(f"  Maximum Privacy Risk: {max_combined:.4f}\n")
        f.write(f"  Minimum Privacy Risk: {min_combined:.4f}\n\n")
        
        f.write(f"Facebook Model:\n")
        f.write(f"  Average Privacy Risk: {avg_fb:.4f}\n")
        f.write(f"  Maximum Privacy Risk: {max_fb:.4f}\n")
        f.write(f"  Minimum Privacy Risk: {min_fb:.4f}\n\n")
        
        f.write(f"Twitter Model:\n")
        f.write(f"  Average Privacy Risk: {avg_tw:.4f}\n")
        f.write(f"  Maximum Privacy Risk: {max_tw:.4f}\n")
        f.write(f"  Minimum Privacy Risk: {min_tw:.4f}\n\n")
        
        f.write("Top 10 Most At-Risk Nodes:\n")
        for i, idx in enumerate(node_indices[:10]):
            original_id = fb_rev_mapping[idx]
            tw_risk_value = tw_risks[idx] if idx < len(tw_risks) else "N/A"
            if tw_risk_value != "N/A":
                f.write(f"{i+1}. Node {original_id}: Combined Risk {combined_risks[idx]:.4f}, "
                       f"FB Risk {fb_risks[idx]:.4f}, TW Risk {tw_risk_value:.4f}\n")
            else:
                f.write(f"{i+1}. Node {original_id}: Combined Risk {combined_risks[idx]:.4f}, "
                       f"FB Risk {fb_risks[idx]:.4f}, TW Risk {tw_risk_value}\n")
        f.write("\n")
        
        # Include feature importance information
        f.write("Feature Importance Analysis:\n")
        f.write("Top Facebook Features:\n")
        for i, feature in enumerate(explanation["facebook"]["important_features"]):
            f.write(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})\n")
        f.write("\n")
        
        f.write("Top Twitter Features:\n")
        for i, feature in enumerate(explanation["twitter"]["important_features"]):
            f.write(f"{i+1}. {feature['feature']} (Importance: {feature['importance']:.4f})\n")
        f.write("\n")
        
        f.write("Combined Important Features:\n")
        for i, feature in enumerate(explanation["combined_important_features"][:5]):
            f.write(f"{i+1}. [{feature['platform']}] {feature['feature']} (Importance: {feature['importance']:.4f})\n")
        f.write("\n")
        
        f.write("Privacy Recommendations:\n")
        f.write("Based on the multi-view analysis, consider limiting the exposure of the following information:\n")
        for feature in explanation["combined_important_features"][:3]:
            f.write(f"- [{feature['platform']}] {feature['feature']}\n")
        f.write("\n")
        
        f.write("Cross-Platform Privacy Insights:\n")
        f.write("1. Review privacy settings on both platforms to ensure consistent protection\n")
        f.write("2. Be aware that information shared on one platform can affect privacy on the other\n")
        f.write("3. Consider using different privacy settings for different social circles\n")
        f.write("4. Limit cross-platform information sharing that could enable identity linkage\n")
        f.write("5. Regularly audit your privacy settings on both platforms\n")
    
    print(f"Multi-view privacy risk analysis completed.")
    print(f"Report saved to {report_path}")
    print(f"Visualization saved to {vis_path}")
    
    return {
        'combined_risks': combined_risks,
        'fb_risks': fb_risks,
        'tw_risks': tw_risks,
        'explanation': explanation
    }

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Facebook data
    print(f"Loading Facebook ego network {args.fb_ego_id}...")
    fb_loader = FacebookDataLoader(data_dir=args.fb_data_dir)
    fb_network = fb_loader.load_ego_network(args.fb_ego_id)
    
    # Load Twitter data
    print(f"Loading Twitter ego network {args.tw_ego_id}...")
    tw_loader = TwitterDataLoader(data_dir=args.tw_data_dir)
    tw_network = tw_loader.load_ego_network(args.tw_ego_id)
    
    # Define sensitive attributes
    sensitive_keywords = [
        'gender', 'birth', 'age', 'location', 'hometown', 'address',
        'education', 'school', 'university', 'work', 'employer',
        'religion', 'political', 'relationship', 'interested_in',
        'family', 'languages', 'email', 'phone'
    ]
    
    # Process Facebook network
    fb_feature_names = fb_network.feature_names
    fb_sensitive_indices = identify_sensitive_attributes(fb_feature_names, sensitive_keywords)
    fb_network = fb_loader.create_privacy_labels([fb_network], fb_sensitive_indices)[0]
    
    # Process Twitter network
    tw_feature_names = tw_network.feature_names
    tw_sensitive_indices = identify_sensitive_attributes(tw_feature_names, sensitive_keywords)
    tw_network = tw_loader.create_privacy_labels([tw_network], tw_sensitive_indices)[0]
    
    print(f"Identified sensitive attributes and created privacy risk labels")
    
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
        conv_type=args.conv_type
    )
    
    # Determine model path
    if args.model_path is None:
        model_path = os.path.join('multiview_output', f'best_model_fb{args.fb_ego_id}_tw{args.tw_ego_id}.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join('multiview_output', f'final_model_fb{args.fb_ego_id}_tw{args.tw_ego_id}.pt')
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
    
    # Move networks to device
    fb_network = fb_network.to(device)
    tw_network = tw_network.to(device)
    
    # Analyze privacy risk
    print(f"Analyzing multi-view privacy risk...")
    results = analyze_multiview_privacy(model, fb_network, tw_network, args.fb_ego_id, args.tw_ego_id, args.output_dir)
    
    # Print summary results to console
    print("\n" + "="*50)
    print(f"MULTI-VIEW PRIVACY RISK ASSESSMENT SUMMARY")
    print(f"Facebook Network: {args.fb_ego_id}, Twitter Network: {args.tw_ego_id}")
    print("="*50)
    
    print(f"\nNetwork-level statistics:")
    
    table_data = []
    table_data.append(["Combined", f"{np.mean(results['combined_risks']):.4f}", 
                      f"{np.max(results['combined_risks']):.4f}", 
                      f"{np.min(results['combined_risks']):.4f}"])
    
    table_data.append(["Facebook", f"{np.mean(results['fb_risks']):.4f}", 
                      f"{np.max(results['fb_risks']):.4f}", 
                      f"{np.min(results['fb_risks']):.4f}"])
    
    # Only include Twitter stats if there are enough nodes
    if len(results['tw_risks']) >= len(results['fb_risks']):
        tw_subset = results['tw_risks'][:len(results['fb_risks'])]
    else:
        tw_subset = results['tw_risks']
    
    table_data.append(["Twitter", f"{np.mean(tw_subset):.4f}", 
                      f"{np.max(tw_subset):.4f}", 
                      f"{np.min(tw_subset):.4f}"])
    
    print(tabulate(table_data, headers=["Model", "Avg Risk", "Max Risk", "Min Risk"], tablefmt="grid"))
    
    # Print top important features
    print("\nTop privacy-sensitive features across platforms:")
    table_data = []
    for i, feature in enumerate(results['explanation']["combined_important_features"][:5]):
        table_data.append([i+1, feature['platform'], feature['feature'], f"{feature['importance']:.4f}"])
    
    print(tabulate(table_data, headers=["Rank", "Platform", "Feature", "Importance"], tablefmt="grid"))
    
    print("\nCross-Platform Privacy Recommendations:")
    print("1. Review privacy settings on both platforms")
    print("2. Limit sharing of sensitive information identified above")
    print("3. Be aware of cross-platform information linkage")
    
    print(f"\nDetailed report saved to: {args.output_dir}/multiview_privacy_report_fb{args.fb_ego_id}_tw{args.tw_ego_id}.txt")
    print(f"Visualization saved to: {args.output_dir}/privacy_comparison_fb{args.fb_ego_id}_tw{args.tw_ego_id}.png")

if __name__ == '__main__':
    main() 