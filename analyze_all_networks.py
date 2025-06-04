import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gnn_privacy_predictor import (
    load_facebook_data, 
    train_iterative_model,
    GCNPrivacyPredictor,
    evaluate_model
)

def get_all_ego_ids():
    """
    Get all ego IDs from the facebook directory by looking for .edges files.
    """
    ego_ids = []
    for filename in os.listdir("facebook"):
        if filename.endswith(".edges"):
            ego_id = filename.split(".")[0]
            ego_ids.append(ego_id)
    return sorted(ego_ids)

def analyze_all_networks(num_iterations=3, privacy_threshold=0.3, intervention_rate=0.6):
    """
    Run the privacy risk analysis on all ego networks in the dataset.
    
    Parameters:
    - num_iterations: Number of iterations for each network (default: 3)
    - privacy_threshold: Threshold for privacy enhancement (default: 0.3)
    - intervention_rate: Proportion of at-risk users to enhance (default: 0.6)
    
    Returns:
    - results_df: DataFrame with results for all networks
    """
    # Get all ego IDs
    ego_ids = get_all_ego_ids()
    print(f"Found {len(ego_ids)} ego networks: {', '.join(ego_ids)}")
    
    # Initialize results storage
    results = []
    
    # Process each ego network
    for ego_id in tqdm(ego_ids, desc="Processing ego networks"):
        print(f"\n{'='*50}")
        print(f"Processing ego network: {ego_id}")
        print(f"{'='*50}")
        
        try:
            # Train the model iteratively on this ego network
            model, performance_history, sensitive_attr_idx, feature_names = train_iterative_model(
                ego_id=ego_id,
                num_iterations=num_iterations,
                privacy_threshold=privacy_threshold,
                intervention_rate=intervention_rate
            )
            
            # Get the final results
            final_mse = performance_history['mse'][-1]
            final_accuracy = performance_history['accuracy'][-1]
            final_f1 = performance_history['f1'][-1]
            final_auc = performance_history['auc'][-1]
            total_utility_loss = sum(performance_history['utility_loss'])
            
            # Load the data again to get network properties
            data, _, _ = load_facebook_data(ego_id, sensitive_attr_idx)
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1] // 2
            
            # Get privacy risk distribution
            model.eval()
            with torch.no_grad():
                predicted_scores = model(data.x, data.edge_index)
                
            high_risk = (predicted_scores < 0.3).sum().item() / num_nodes * 100
            medium_risk = ((predicted_scores >= 0.3) & (predicted_scores < 0.7)).sum().item() / num_nodes * 100
            low_risk = (predicted_scores >= 0.7).sum().item() / num_nodes * 100
            
            # Save results
            results.append({
                'ego_id': ego_id,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'sensitive_attribute': feature_names[sensitive_attr_idx],
                'final_mse': final_mse,
                'final_accuracy': final_accuracy,
                'final_f1': final_f1, 
                'final_auc': final_auc,
                'total_utility_loss': total_utility_loss,
                'high_risk_percent': high_risk,
                'medium_risk_percent': medium_risk,
                'low_risk_percent': low_risk
            })
            
            # Save the model and performance history for this ego network
            torch.save(model.state_dict(), f"model_ego_{ego_id}.pt")
            
            # Visualize results for this network
            visualize_network_results(ego_id, model, data, performance_history, sensitive_attr_idx, feature_names)
            
        except Exception as e:
            print(f"Error processing ego network {ego_id}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv("all_networks_results.csv", index=False)
    
    return results_df

def visualize_network_results(ego_id, model, data, performance_history, sensitive_attr_idx, feature_names):
    """
    Create visualizations for a single network's results.
    
    Parameters:
    - ego_id: ID of the ego network
    - model: Trained GCNPrivacyPredictor model
    - data: PyTorch Geometric Data object
    - performance_history: Dictionary with performance metrics
    - sensitive_attr_idx: Index of the sensitive attribute
    - feature_names: Names of the features
    """
    # Create a directory for this ego network's results
    os.makedirs(f"results_ego_{ego_id}", exist_ok=True)
    
    # Plot performance metrics
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
    plt.savefig(f"results_ego_{ego_id}/performance_metrics.png")
    plt.close()
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predicted_scores = model(data.x, data.edge_index)
    
    # Plot privacy risk distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_scores.numpy(), bins=20, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.axvline(x=0.3, color='red', linestyle='--', label='High Risk Threshold')
    plt.axvline(x=0.7, color='green', linestyle='--', label='Low Risk Threshold')
    plt.xlabel('Privacy Score (lower = higher risk)')
    plt.ylabel('Number of Users')
    plt.title(f'Privacy Risk Distribution for Ego Network {ego_id}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results_ego_{ego_id}/privacy_distribution.png")
    plt.close()

def visualize_aggregate_results(results_df):
    """
    Create visualizations for the aggregate results across all networks.
    
    Parameters:
    - results_df: DataFrame with results for all networks
    """
    # Plot network size vs. privacy metrics
    plt.figure(figsize=(15, 10))
    
    # Network size vs. accuracy
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['num_nodes'], results_df['final_accuracy'], alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Final Accuracy')
    plt.title('Network Size vs. Accuracy')
    for i, txt in enumerate(results_df['ego_id']):
        plt.annotate(txt, (results_df['num_nodes'].iloc[i], results_df['final_accuracy'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    plt.grid(True, alpha=0.3)
    
    # Network size vs. F1 score
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['num_nodes'], results_df['final_f1'], alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Final F1 Score')
    plt.title('Network Size vs. F1 Score')
    for i, txt in enumerate(results_df['ego_id']):
        plt.annotate(txt, (results_df['num_nodes'].iloc[i], results_df['final_f1'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    plt.grid(True, alpha=0.3)
    
    # Network size vs. AUC
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['num_nodes'], results_df['final_auc'], alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Final AUC')
    plt.title('Network Size vs. AUC')
    for i, txt in enumerate(results_df['ego_id']):
        plt.annotate(txt, (results_df['num_nodes'].iloc[i], results_df['final_auc'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    plt.grid(True, alpha=0.3)
    
    # Network size vs. utility loss
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['num_nodes'], results_df['total_utility_loss'], alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Total Utility Loss')
    plt.title('Network Size vs. Utility Loss')
    for i, txt in enumerate(results_df['ego_id']):
        plt.annotate(txt, (results_df['num_nodes'].iloc[i], results_df['total_utility_loss'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("aggregate_network_size_vs_metrics.png")
    plt.close()
    
    # Plot risk distribution across networks
    plt.figure(figsize=(12, 6))
    
    # Sort by high risk percentage
    sorted_df = results_df.sort_values(by='high_risk_percent', ascending=False)
    
    # Create stacked bar chart
    bars = plt.bar(sorted_df['ego_id'], sorted_df['high_risk_percent'], color='red', label='High Risk')
    bars = plt.bar(sorted_df['ego_id'], sorted_df['medium_risk_percent'], 
                  bottom=sorted_df['high_risk_percent'], color='yellow', label='Medium Risk')
    bars = plt.bar(sorted_df['ego_id'], sorted_df['low_risk_percent'],
                  bottom=sorted_df['high_risk_percent'] + sorted_df['medium_risk_percent'], 
                  color='green', label='Low Risk')
    
    plt.xlabel('Ego Network ID')
    plt.ylabel('Percentage of Users')
    plt.title('Privacy Risk Distribution Across Networks')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("aggregate_risk_distribution.png")
    plt.close()
    
    # Create a summary table as an image
    plt.figure(figsize=(12, len(results_df) * 0.5 + 1))
    plt.axis('off')
    
    # Create table data
    table_data = [
        ['Ego ID', 'Nodes', 'Edges', 'Accuracy', 'F1', 'AUC', 'Utility Loss', 'High Risk %']
    ]
    
    for _, row in results_df.iterrows():
        table_data.append([
            row['ego_id'],
            str(row['num_nodes']),
            str(row['num_edges']),
            f"{row['final_accuracy']:.3f}",
            f"{row['final_f1']:.3f}",
            f"{row['final_auc']:.3f}",
            f"{row['total_utility_loss']:.3f}",
            f"{row['high_risk_percent']:.1f}%"
        ])
    
    # Create table
    table = plt.table(cellText=table_data, colWidths=[0.1] * 8, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title
    plt.title('Summary of Results Across All Ego Networks', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig("results_summary_table.png", bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    print("Analyzing all ego networks in the Facebook dataset...")
    
    # Run analysis on all networks
    results_df = analyze_all_networks(
        num_iterations=3,  # Reduced for faster execution
        privacy_threshold=0.3,
        intervention_rate=0.6
    )
    
    # Create aggregate visualizations
    print("\nCreating aggregate visualizations...")
    visualize_aggregate_results(results_df)
    
    print("\nAnalysis complete!")
    print(f"Results saved to 'all_networks_results.csv' and individual network results in 'results_ego_*' directories")
    print("Aggregate visualizations: 'aggregate_network_size_vs_metrics.png', 'aggregate_risk_distribution.png', and 'results_summary_table.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total number of networks analyzed: {len(results_df)}")
    
    if not results_df.empty:
        print(f"Average accuracy across networks: {results_df['final_accuracy'].mean():.3f}")
        print(f"Average F1 score across networks: {results_df['final_f1'].mean():.3f}")
        print(f"Average AUC across networks: {results_df['final_auc'].mean():.3f}")
        print(f"Average utility loss across networks: {results_df['total_utility_loss'].mean():.3f}")
        print(f"Average percentage of high-risk users: {results_df['high_risk_percent'].mean():.1f}%")
        
        # Network with highest risk
        highest_risk_network = results_df.loc[results_df['high_risk_percent'].idxmax()]
        print(f"\nNetwork with highest privacy risk: {highest_risk_network['ego_id']} "
              f"({highest_risk_network['high_risk_percent']:.1f}% high-risk users)")
        
        # Network with best model performance
        best_auc_network = results_df.loc[results_df['final_auc'].idxmax()]
        print(f"Network with best model performance (AUC): {best_auc_network['ego_id']} "
              f"(AUC: {best_auc_network['final_auc']:.3f})") 