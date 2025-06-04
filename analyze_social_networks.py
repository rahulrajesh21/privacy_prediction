import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
import traceback
from collections import defaultdict
import platform
import gc
import shutil
from datetime import datetime

# Set this to True to process only one network per dataset for testing
DEBUG_MODE = False
# Set a smaller batch size if experiencing memory issues
BATCH_SIZE = 50
# Output directory for all results
OUTPUT_DIR = "privacy_analysis_results"

class SocialNetworkAnalyzer:
    """
    A class to analyze social network data from Facebook and Twitter datasets.
    """
    def __init__(self):
        # Create output directory
        if os.path.exists(OUTPUT_DIR):
            print(f"Cleaning previous output directory: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")
        
        # Create subdirectories for visualizations
        os.makedirs(os.path.join(OUTPUT_DIR, "facebook"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "twitter"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "summary"), exist_ok=True)
        
        # Check for MPS (Metal Performance Shaders) for Mac M-series chips
        if torch.backends.mps.is_available() and platform.system() == 'Darwin' and 'arm' in platform.machine():
            self.device = torch.device("mps")
            print("Using Mac M-series GPU (MPS)")
        # Check for CUDA
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # Fallback to CPU
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computations")
        
        # Load PyTorch Geometric only if needed
        try:
            from torch_geometric.nn import GCNConv
            from torch_geometric.data import Data
            self.torch_geometric_available = True
            self.Data = Data
            self.GCNConv = GCNConv
            print("PyTorch Geometric is available")
        except ImportError:
            print("Warning: torch-geometric not available. Limited functionality.")
            self.torch_geometric_available = False
    
    def get_ego_ids(self, dataset_name):
        """
        Get all ego IDs from the specified dataset by looking for .edges files.
        
        Parameters:
        - dataset_name: 'facebook' or 'twitter'
        
        Returns:
        - sorted list of ego IDs
        """
        if not os.path.exists(dataset_name):
            print(f"Error: {dataset_name} directory not found")
            return []
            
        ego_ids = []
        for filename in os.listdir(dataset_name):
            if filename.endswith(".edges"):
                ego_id = filename.split(".")[0]
                ego_ids.append(ego_id)
        
        ego_ids = sorted(ego_ids)
        
        # If in debug mode, only use the first ego network
        if DEBUG_MODE:
            if ego_ids:
                return [ego_ids[0]]
        
        return ego_ids
    
    def load_network_data(self, dataset_name, ego_id, sensitive_attr_idx=None):
        """
        Load data for a specific ego network.
        
        Parameters:
        - dataset_name: 'facebook' or 'twitter'
        - ego_id: ID of the ego user
        - sensitive_attr_idx: Index of the feature to use as sensitive attribute (or None)
        
        Returns:
        - data: Dict containing network data or PyTorch Geometric Data object
        - sensitive_attr_idx: Index of the sensitive attribute
        - feature_names: Names of all features
        """
        print(f"\nLoading {dataset_name} network {ego_id}...")
        base_dir = dataset_name
        
        # Check for required files
        required_files = [
            os.path.join(base_dir, f"{ego_id}.edges"),
            os.path.join(base_dir, f"{ego_id}.feat"),
            os.path.join(base_dir, f"{ego_id}.featnames")
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load edges
        edges_file = os.path.join(base_dir, f"{ego_id}.edges")
        print(f"  Loading edges from {edges_file}...")
        edges_df = pd.read_csv(edges_file, sep=' ', header=None, names=['source', 'target'])
        print(f"  Loaded {len(edges_df)} edges")
        
        # Load features
        feat_file = os.path.join(base_dir, f"{ego_id}.feat")
        print(f"  Loading features from {feat_file}...")
        feat_df = pd.read_csv(feat_file, sep=' ', header=None)
        print(f"  Loaded features for {feat_df.shape[0]} nodes with {feat_df.shape[1]-1} attributes")
        
        # Load feature names
        featnames_file = os.path.join(base_dir, f"{ego_id}.featnames")
        print(f"  Loading feature names from {featnames_file}...")
        feature_names = []
        with open(featnames_file, 'r') as f:
            for line in f:
                if ' ' in line.strip():  # Make sure we can split the line
                    idx, name = line.strip().split(' ', 1)
                    feature_names.append(name)
        print(f"  Loaded {len(feature_names)} feature names")
        
        # Convert node IDs and edges to proper format
        print("  Processing network data...")
        node_ids = feat_df[0].values
        features = feat_df.iloc[:, 1:].values
        
        # Create node ID mapping
        node_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
        
        # Map edge IDs
        print("  Building edge index...")
        edge_index = []
        for _, row in edges_df.iterrows():
            if row['source'] in node_map and row['target'] in node_map:
                edge_index.append([node_map[row['source']], node_map[row['target']]])
                # Add the reverse edge for undirected graph
                edge_index.append([node_map[row['target']], node_map[row['source']]])
        
        # Convert to tensors and move to the selected device
        print(f"  Creating tensors on {self.device}...")
        try:
            # Try with GPU first
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
            x_tensor = torch.tensor(features, dtype=torch.float, device=self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Fall back to CPU if GPU memory error
            print(f"  GPU memory error: {str(e)}")
            print("  Falling back to CPU...")
            self.device = torch.device("cpu")
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
            x_tensor = torch.tensor(features, dtype=torch.float)
        
        # If no sensitive attribute specified, randomly select one
        if sensitive_attr_idx is None:
            print("  Selecting sensitive attribute...")
            # Choose an attribute with some variability (not all 0s or 1s)
            candidates = []
            for i in range(x_tensor.shape[1]):
                try:
                    unique_vals = torch.unique(x_tensor[:, i])
                    if len(unique_vals) > 1 and len(unique_vals) < x_tensor.shape[0] * 0.8:
                        candidates.append(i)
                except:
                    continue
            
            if candidates:
                sensitive_attr_idx = np.random.choice(candidates)
            else:
                sensitive_attr_idx = np.random.randint(0, x_tensor.shape[1] - 1)
            print(f"  Selected attribute index {sensitive_attr_idx}")
        
        # Create result dictionary
        result = {
            'node_ids': node_ids,
            'node_map': node_map,
            'edges_df': edges_df,
            'features': features,
            'edge_index': edge_index_tensor,
            'x': x_tensor,
            'num_nodes': len(node_ids),
            'num_edges': len(edges_df),
            'num_features': features.shape[1]
        }
        
        # Create PyTorch Geometric Data object if available
        if self.torch_geometric_available:
            print("  Creating PyTorch Geometric Data object...")
            result['pyg_data'] = self.Data(x=x_tensor, edge_index=edge_index_tensor)
        
        print(f"  Successfully loaded {dataset_name} network {ego_id}")
        return result, sensitive_attr_idx, feature_names
    
    def calculate_network_metrics(self, data):
        """
        Calculate basic network metrics.
        
        Parameters:
        - data: Dict containing network data
        
        Returns:
        - metrics: Dict with calculated metrics
        """
        print("  Calculating network metrics...")
        # Calculate degree distribution
        edge_index = data['edge_index']
        num_nodes = data['num_nodes']
        
        # Move to CPU for non-GPU operations if needed
        if edge_index.device.type != 'cpu':
            print("    Moving data to CPU for metrics calculation...")
            edge_index_cpu = edge_index.cpu()
        else:
            edge_index_cpu = edge_index
        
        # Count degrees for each node
        print("    Computing degree distribution...")
        degrees = defaultdict(int)
        for i in range(edge_index_cpu.shape[1]):
            node = edge_index_cpu[0, i].item()
            degrees[node] += 1
        
        # Calculate metrics
        degree_values = list(degrees.values())
        avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0
        max_degree = max(degree_values) if degree_values else 0
        
        # Calculate density
        print("    Computing network density...")
        possible_edges = num_nodes * (num_nodes - 1) / 2
        actual_edges = data['num_edges']
        density = actual_edges / possible_edges if possible_edges > 0 else 0
        
        metrics = {
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'density': density,
            'degree_distribution': degree_values
        }
        
        print(f"    Avg degree: {avg_degree:.2f}, Max degree: {max_degree}, Density: {density:.6f}")
        return metrics
    
    def analyze_privacy_risk(self, data, sensitive_attr_idx):
        """
        Perform basic privacy risk assessment based on network structure.
        
        Parameters:
        - data: Dict containing network data
        - sensitive_attr_idx: Index of the sensitive attribute
        
        Returns:
        - risk_scores: List of privacy risk scores for each node
        - high_risk_percentage: Percentage of nodes at high risk
        """
        print("  Analyzing privacy risk...")
        x = data['x']
        edge_index = data['edge_index']
        num_nodes = data['num_nodes']
        
        # Move tensors to CPU for easier processing if they're on GPU
        if x.device.type != 'cpu':
            print("    Moving data to CPU for risk analysis...")
            x_cpu = x.cpu()
            edge_index_cpu = edge_index.cpu()
        else:
            x_cpu = x
            edge_index_cpu = edge_index
        
        # Calculate privacy risk for each node based on neighborhood homogeneity
        risk_scores = []
        
        # Process in batches to improve performance
        print(f"    Processing {num_nodes} nodes in batches of {BATCH_SIZE}...")
        
        progress_bar = tqdm(total=num_nodes, desc="    Analyzing nodes", leave=False)
        
        for batch_start in range(0, num_nodes, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_nodes)
            batch_size = batch_end - batch_start
            
            # For each node in this batch, calculate risk
            for node_idx in range(batch_start, batch_end):
                # Find neighbors
                neighbors = []
                for i in range(edge_index_cpu.shape[1]):
                    if edge_index_cpu[0, i].item() == node_idx:
                        neighbors.append(edge_index_cpu[1, i].item())
                
                if not neighbors:
                    # No neighbors means no risk of inference
                    risk_scores.append(0.0)
                    progress_bar.update(1)
                    continue
                
                # Get the sensitive attribute value for this node
                sensitive_value = x_cpu[node_idx, sensitive_attr_idx].item()
                
                # Count neighbors with the same sensitive value
                matching_neighbors = sum(1 for n in neighbors if x_cpu[n, sensitive_attr_idx].item() == sensitive_value)
                
                # Calculate risk score (higher score = higher risk)
                if len(neighbors) > 0:
                    risk_score = matching_neighbors / len(neighbors)
                else:
                    risk_score = 0.0
                
                risk_scores.append(risk_score)
                progress_bar.update(1)
            
            # Periodically clear memory if using GPU
            if x.device.type != 'cpu':
                torch.cuda.empty_cache()
        
        progress_bar.close()
        
        # Calculate percentage of high-risk nodes (risk score > 0.7)
        high_risk_percentage = sum(1 for score in risk_scores if score > 0.7) / num_nodes * 100
        print(f"    High risk percentage: {high_risk_percentage:.2f}%")
        
        return risk_scores, high_risk_percentage
    
    def analyze_all_networks(self, dataset_names=['facebook', 'twitter']):
        """
        Run analysis on all networks in the specified datasets.
        
        Parameters:
        - dataset_names: List of dataset names to analyze
        
        Returns:
        - results_df: DataFrame with results for all networks
        """
        results = []
        
        for dataset_name in dataset_names:
            print(f"\nAnalyzing {dataset_name} networks...")
            ego_ids = self.get_ego_ids(dataset_name)
            
            if not ego_ids:
                print(f"No ego networks found in {dataset_name} directory.")
                continue
                
            print(f"Found {len(ego_ids)} ego networks in {dataset_name}: {', '.join(ego_ids[:5])}...")
            
            # Process each ego network
            for i, ego_id in enumerate(ego_ids):
                print(f"Processing {dataset_name} network {i+1}/{len(ego_ids)}: {ego_id}")
                
                try:
                    start_time = time.time()
                    
                    # Load network data
                    network_data, sensitive_attr_idx, feature_names = self.load_network_data(
                        dataset_name, ego_id
                    )
                    
                    # Calculate network metrics
                    metrics = self.calculate_network_metrics(network_data)
                    
                    # Analyze privacy risk
                    risk_scores, high_risk_percentage = self.analyze_privacy_risk(
                        network_data, sensitive_attr_idx
                    )
                    
                    # Create result entry
                    result = {
                        'dataset': dataset_name,
                        'ego_id': ego_id,
                        'num_nodes': network_data['num_nodes'],
                        'num_edges': network_data['num_edges'],
                        'num_features': network_data['num_features'],
                        'sensitive_attribute': feature_names[sensitive_attr_idx] if sensitive_attr_idx < len(feature_names) else "unknown",
                        'avg_degree': metrics['avg_degree'],
                        'max_degree': metrics['max_degree'],
                        'density': metrics['density'],
                        'high_risk_percentage': high_risk_percentage,
                        'avg_risk_score': np.mean(risk_scores)
                    }
                    
                    results.append(result)
                    
                    # Visualize this network
                    print("  Creating visualizations...")
                    self.visualize_network(
                        dataset_name, 
                        ego_id, 
                        network_data, 
                        risk_scores, 
                        sensitive_attr_idx,
                        feature_names
                    )
                    
                    # Clean up to free memory
                    del network_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    elapsed_time = time.time() - start_time
                    print(f"  Completed in {elapsed_time:.1f} seconds\n")
                    
                except Exception as e:
                    print(f"Error processing {dataset_name} network {ego_id}:")
                    traceback.print_exc()
                    print(f"Continuing with next network...\n")
                    continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # No need to save partial results as they will all be in the output directory
        
        return results_df
    
    def visualize_network(self, dataset_name, ego_id, data, risk_scores, sensitive_attr_idx, feature_names):
        """
        Create visualizations for a single network.
        
        Parameters:
        - dataset_name: Name of the dataset ('facebook' or 'twitter')
        - ego_id: ID of the ego network
        - data: Dict containing network data
        - risk_scores: List of privacy risk scores
        - sensitive_attr_idx: Index of the sensitive attribute
        - feature_names: Names of the features
        """
        # Create directory for visualizations within the output directory
        output_dir = os.path.join(OUTPUT_DIR, dataset_name, f"network_{ego_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data to CPU if it's on GPU
        edge_index = data['edge_index'].cpu() if data['edge_index'].device.type != 'cpu' else data['edge_index']
        x = data['x'].cpu() if data['x'].device.type != 'cpu' else data['x']
        
        # Plot degree distribution
        plt.figure(figsize=(10, 6))
        plt.hist(edge_index[0].numpy(), bins=30, alpha=0.7, color='blue')
        plt.xlabel('Node ID')
        plt.ylabel('Degree')
        plt.title(f'Degree Distribution for {dataset_name} Network {ego_id}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/degree_distribution.png")
        plt.close()
        
        # Plot risk score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(risk_scores, bins=20, color='red', alpha=0.7)
        plt.axvline(x=0.7, color='black', linestyle='--', label='High Risk Threshold')
        plt.xlabel('Privacy Risk Score (higher = higher risk)')
        plt.ylabel('Number of Users')
        plt.title(f'Privacy Risk Distribution for {dataset_name} Network {ego_id}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/privacy_risk_distribution.png")
        plt.close()
        
        # Plot sensitive attribute distribution
        try:
            sensitive_values = x[:, sensitive_attr_idx].numpy()
            unique_values, counts = np.unique(sensitive_values, return_counts=True)
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(unique_values)), counts, color='green')
            plt.xlabel('Attribute Value')
            plt.ylabel('Count')
            plt.title(f'Distribution of Sensitive Attribute: {feature_names[sensitive_attr_idx]}')
            plt.xticks(range(len(unique_values)), [f"{v:.1f}" for v in unique_values])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sensitive_attribute_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting sensitive attribute distribution: {str(e)}")
    
    def visualize_aggregate_results(self, results_df):
        """
        Create visualizations for the aggregate results across all networks.
        
        Parameters:
        - results_df: DataFrame with results for all networks
        """
        if results_df.empty:
            print("No results to visualize.")
            return
            
        # Directory for summary visualizations
        summary_dir = os.path.join(OUTPUT_DIR, "summary")
        
        # Create comparison between Facebook and Twitter networks
        if 'facebook' in results_df['dataset'].values and 'twitter' in results_df['dataset'].values:
            # Compare network size
            plt.figure(figsize=(12, 8))
            
            # Network size comparison
            plt.subplot(2, 2, 1)
            fb_sizes = results_df[results_df['dataset'] == 'facebook']['num_nodes']
            tw_sizes = results_df[results_df['dataset'] == 'twitter']['num_nodes']
            plt.boxplot([fb_sizes, tw_sizes], labels=['Facebook', 'Twitter'])
            plt.ylabel('Number of Nodes')
            plt.title('Network Size Comparison')
            plt.grid(alpha=0.3)
            
            # Density comparison
            plt.subplot(2, 2, 2)
            fb_density = results_df[results_df['dataset'] == 'facebook']['density']
            tw_density = results_df[results_df['dataset'] == 'twitter']['density']
            plt.boxplot([fb_density, tw_density], labels=['Facebook', 'Twitter'])
            plt.ylabel('Network Density')
            plt.title('Network Density Comparison')
            plt.grid(alpha=0.3)
            
            # Average degree comparison
            plt.subplot(2, 2, 3)
            fb_degree = results_df[results_df['dataset'] == 'facebook']['avg_degree']
            tw_degree = results_df[results_df['dataset'] == 'twitter']['avg_degree']
            plt.boxplot([fb_degree, tw_degree], labels=['Facebook', 'Twitter'])
            plt.ylabel('Average Degree')
            plt.title('Average Degree Comparison')
            plt.grid(alpha=0.3)
            
            # Privacy risk comparison
            plt.subplot(2, 2, 4)
            fb_risk = results_df[results_df['dataset'] == 'facebook']['high_risk_percentage']
            tw_risk = results_df[results_df['dataset'] == 'twitter']['high_risk_percentage']
            plt.boxplot([fb_risk, tw_risk], labels=['Facebook', 'Twitter'])
            plt.ylabel('High Risk Percentage')
            plt.title('Privacy Risk Comparison')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{summary_dir}/facebook_vs_twitter_comparison.png")
            plt.close()
        
        # Network size vs. privacy risk
        plt.figure(figsize=(10, 6))
        for dataset in results_df['dataset'].unique():
            subset = results_df[results_df['dataset'] == dataset]
            plt.scatter(subset['num_nodes'], subset['high_risk_percentage'], 
                       label=dataset.capitalize(), alpha=0.7)
            
            # Add ego IDs as annotations
            for i, row in subset.iterrows():
                plt.annotate(row['ego_id'], 
                            (row['num_nodes'], row['high_risk_percentage']),
                            textcoords="offset points", 
                            xytext=(0, 5), 
                            ha='center',
                            fontsize=8)
        
        plt.xlabel('Number of Nodes')
        plt.ylabel('High Risk Percentage')
        plt.title('Network Size vs. Privacy Risk')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/network_size_vs_privacy_risk.png")
        plt.close()
        
        # Network density vs. privacy risk
        plt.figure(figsize=(10, 6))
        for dataset in results_df['dataset'].unique():
            subset = results_df[results_df['dataset'] == dataset]
            plt.scatter(subset['density'], subset['high_risk_percentage'], 
                       label=dataset.capitalize(), alpha=0.7)
        
        plt.xlabel('Network Density')
        plt.ylabel('High Risk Percentage')
        plt.title('Network Density vs. Privacy Risk')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/network_density_vs_privacy_risk.png")
        plt.close()
        
        # Create a summary table as an image
        plt.figure(figsize=(14, len(results_df) * 0.3 + 2))
        plt.axis('off')
        
        # Group by dataset and calculate statistics
        stats = results_df.groupby('dataset').agg({
            'num_nodes': ['mean', 'std', 'min', 'max'],
            'num_edges': ['mean', 'std', 'min', 'max'],
            'density': ['mean', 'std'],
            'high_risk_percentage': ['mean', 'std', 'min', 'max'],
            'ego_id': 'count'
        }).reset_index()
        
        # Rename count to num_networks
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        stats = stats.rename(columns={'ego_id_count': 'num_networks'})
        
        # Format for table display
        table_data = [
            ['Dataset', 'Networks', 'Avg Nodes', 'Avg Edges', 'Avg Density', 'Avg High Risk %']
        ]
        
        for _, row in stats.iterrows():
            table_data.append([
                row['dataset'].capitalize(),
                str(int(row['num_networks'])),
                f"{row['num_nodes_mean']:.1f} ± {row['num_nodes_std']:.1f}",
                f"{row['num_edges_mean']:.1f} ± {row['num_edges_std']:.1f}",
                f"{row['density_mean']:.4f} ± {row['density_std']:.4f}",
                f"{row['high_risk_percentage_mean']:.1f}% ± {row['high_risk_percentage_std']:.1f}%"
            ])
        
        # Create table
        table = plt.table(cellText=table_data, colWidths=[0.15] * 6, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add title
        plt.title('Summary Statistics by Dataset', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/summary_statistics_table.png", bbox_inches='tight', dpi=200)
        plt.close()
        
        # Individual network summary
        plt.figure(figsize=(15, len(results_df) * 0.3 + 2))
        plt.axis('off')
        
        # Create table data for individual networks
        table_data = [
            ['Dataset', 'Ego ID', 'Nodes', 'Edges', 'Density', 'Avg Degree', 'High Risk %']
        ]
        
        sorted_df = results_df.sort_values(by=['dataset', 'num_nodes'], ascending=[True, False])
        
        for _, row in sorted_df.iterrows():
            table_data.append([
                row['dataset'].capitalize(),
                row['ego_id'],
                str(row['num_nodes']),
                str(row['num_edges']),
                f"{row['density']:.4f}",
                f"{row['avg_degree']:.1f}",
                f"{row['high_risk_percentage']:.1f}%"
            ])
        
        # Create table
        table = plt.table(cellText=table_data, colWidths=[0.12] * 7, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        
        # Add title
        plt.title('Individual Network Summary', pad=20, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{summary_dir}/individual_networks_table.png", bbox_inches='tight', dpi=200)
        plt.close()
        
        # Save the results dataframe to CSV
        results_df.to_csv(f"{OUTPUT_DIR}/all_networks_results.csv", index=False)
        
        # Create an HTML report
        self.create_html_report(results_df, summary_dir)
    
    def create_html_report(self, results_df, summary_dir):
        """
        Create an HTML report with all the results.
        
        Parameters:
        - results_df: DataFrame with results for all networks
        - summary_dir: Directory for summary files
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Social Network Privacy Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .summary {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                .summary-item {{ margin: 10px; text-align: center; }}
                table {{ border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .networks {{ display: flex; flex-wrap: wrap; }}
                .network {{ margin: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Social Network Privacy Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="summary">
                <div class="summary-item">
                    <h3>Network Size vs. Privacy Risk</h3>
                    <img src="network_size_vs_privacy_risk.png" alt="Network Size vs. Privacy Risk">
                </div>
                <div class="summary-item">
                    <h3>Network Density vs. Privacy Risk</h3>
                    <img src="network_density_vs_privacy_risk.png" alt="Network Density vs. Privacy Risk">
                </div>
            </div>
            
            <h3>Dataset Comparison</h3>
            <img src="facebook_vs_twitter_comparison.png" alt="Facebook vs Twitter Comparison">
            
            <h3>Summary Statistics</h3>
            <img src="summary_statistics_table.png" alt="Summary Statistics">
            
            <h3>Individual Network Details</h3>
            <img src="individual_networks_table.png" alt="Individual Networks Table">
            
            <h2>All Networks Results</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Ego ID</th>
                    <th>Nodes</th>
                    <th>Edges</th>
                    <th>Density</th>
                    <th>Avg Degree</th>
                    <th>High Risk %</th>
                </tr>
        """
        
        # Add rows for each network
        for _, row in results_df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['dataset'].capitalize()}</td>
                    <td>{row['ego_id']}</td>
                    <td>{row['num_nodes']}</td>
                    <td>{row['num_edges']}</td>
                    <td>{row['density']:.4f}</td>
                    <td>{row['avg_degree']:.1f}</td>
                    <td>{row['high_risk_percentage']:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Network Visualizations</h2>
        """
        
        # Add sections for each dataset
        for dataset in sorted(results_df['dataset'].unique()):
            html_content += f"""
            <h3>{dataset.capitalize()} Networks</h3>
            <div class="networks">
            """
            
            # Add links to individual network visualizations
            dataset_networks = results_df[results_df['dataset'] == dataset]
            for _, row in dataset_networks.iterrows():
                ego_id = row['ego_id']
                html_content += f"""
                <div class="network">
                    <h4>Network {ego_id}</h4>
                    <p>Nodes: {row['num_nodes']} | Edges: {row['num_edges']} | High Risk: {row['high_risk_percentage']:.1f}%</p>
                    <a href="../{dataset}/network_{ego_id}/privacy_risk_distribution.png" target="_blank">
                        <img src="../{dataset}/network_{ego_id}/privacy_risk_distribution.png" width="300" alt="Privacy Risk Distribution">
                    </a>
                </div>
                """
            
            html_content += """
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML file
        with open(os.path.join(summary_dir, "report.html"), "w") as f:
            f.write(html_content)
            
        print(f"HTML report saved to {os.path.join(summary_dir, 'report.html')}")

if __name__ == "__main__":
    print("=== Social Network Privacy Analysis ===")
    start_time = time.time()
    
    # Create analyzer
    analyzer = SocialNetworkAnalyzer()
    
    # Process all networks
    results = analyzer.analyze_all_networks(['facebook', 'twitter'])
    
    # Create aggregate visualizations
    print("\nCreating aggregate visualizations and final report...")
    analyzer.visualize_aggregate_results(results)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Results saved to '{OUTPUT_DIR}' directory")
    print(f"Full report available at '{OUTPUT_DIR}/summary/report.html'")
    
    # Print dataset comparison
    if not results.empty:
        for dataset in results['dataset'].unique():
            subset = results[results['dataset'] == dataset]
            print(f"\n{dataset.capitalize()} Networks:")
            print(f"  Number of networks: {len(subset)}")
            print(f"  Average number of nodes: {subset['num_nodes'].mean():.1f}")
            print(f"  Average number of edges: {subset['num_edges'].mean():.1f}")
            print(f"  Average network density: {subset['density'].mean():.4f}")
            print(f"  Average percentage of high-risk users: {subset['high_risk_percentage'].mean():.1f}%")
        
        # Identify most risky networks
        for dataset in results['dataset'].unique():
            subset = results[results['dataset'] == dataset]
            if not subset.empty:
                highest_risk = subset.loc[subset['high_risk_percentage'].idxmax()]
                print(f"\nMost privacy-sensitive {dataset} network: {highest_risk['ego_id']} "
                    f"({highest_risk['high_risk_percentage']:.1f}% high-risk users)")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.1f} seconds") 