import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm

class FacebookDataLoader:
    def __init__(self, data_dir='facebook'):
        self.data_dir = data_dir
        self.ego_ids = self._get_ego_ids()
        
    def _get_ego_ids(self):
        """Get all ego network IDs from the dataset directory."""
        files = os.listdir(self.data_dir)
        ego_ids = set()
        for file in files:
            if file.endswith('.edges'):
                ego_ids.add(file.split('.')[0])
        return sorted(list(ego_ids))
    
    def load_ego_network(self, ego_id):
        """Load a single ego network with the given ID."""
        # Load edges
        edges_file = os.path.join(self.data_dir, f"{ego_id}.edges")
        edges_df = pd.read_csv(edges_file, sep=' ', header=None, names=['source', 'target'])
        
        # Load node features
        feat_file = os.path.join(self.data_dir, f"{ego_id}.feat")
        feat_df = pd.read_csv(feat_file, sep=' ', header=None)
        node_ids = feat_df[0].values
        features = feat_df.iloc[:, 1:-1].values  # Exclude node ID and last column
        
        # Load feature names for interpretability
        featnames_file = os.path.join(self.data_dir, f"{ego_id}.featnames")
        featnames_df = pd.read_csv(featnames_file, sep=' ', header=None, names=['id', 'name'])
        feature_names = featnames_df['name'].tolist()
        
        # Load ego's own features
        ego_feat_file = os.path.join(self.data_dir, f"{ego_id}.egofeat")
        ego_feat = pd.read_csv(ego_feat_file, sep=' ', header=None).values.flatten()[:-1]  # Exclude last column
        
        # Load circles (communities)
        circles_file = os.path.join(self.data_dir, f"{ego_id}.circles")
        circles = {}
        with open(circles_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                circle_name = parts[0]
                members = [int(x) for x in parts[1:]]
                circles[circle_name] = members
        
        # Create node ID mapping (original ID -> consecutive index)
        # Add ego node (not in feat file)
        all_nodes = np.append(node_ids, int(ego_id))
        node_mapping = {int(node): idx for idx, node in enumerate(all_nodes)}
        
        # Map edges to new node indices
        edge_index = []
        for _, row in edges_df.iterrows():
            source = node_mapping[row['source']]
            target = node_mapping[row['target']]
            edge_index.append([source, target])
            edge_index.append([target, source])  # Add reverse edge for undirected graph
        
        # Add connections from ego to all nodes
        ego_idx = node_mapping[int(ego_id)]
        for node_id in node_ids:
            node_idx = node_mapping[node_id]
            edge_index.append([ego_idx, node_idx])
            edge_index.append([node_idx, ego_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Combine ego features with other node features
        all_features = np.vstack([features, ego_feat])
        x = torch.tensor(all_features, dtype=torch.float)
        
        # Create circles mask for each node (which circle it belongs to)
        circles_mask = np.zeros((len(all_nodes), len(circles)), dtype=np.float32)
        for circle_idx, (_, members) in enumerate(circles.items()):
            for member in members:
                if member in node_mapping:
                    node_idx = node_mapping[member]
                    circles_mask[node_idx, circle_idx] = 1.0
        
        circles_mask = torch.tensor(circles_mask, dtype=torch.float)
        
        # Create PyG data object
        data = Data(
            x=x,
            edge_index=edge_index,
            circles=circles_mask,
            node_mapping=node_mapping,
            feature_names=feature_names,
            ego_id=int(ego_id)
        )
        
        return data
    
    def load_all_networks(self):
        """Load all ego networks in the dataset."""
        all_networks = []
        for ego_id in tqdm(self.ego_ids, desc="Loading ego networks"):
            try:
                network = self.load_ego_network(ego_id)
                all_networks.append(network)
            except Exception as e:
                print(f"Error loading ego network {ego_id}: {e}")
        
        return all_networks

    def create_privacy_labels(self, data_list, sensitive_attributes):
        """
        Create privacy sensitivity labels based on sensitive attributes.
        
        Args:
            data_list: List of PyG Data objects
            sensitive_attributes: List of feature indices considered sensitive
            
        Returns:
            Updated data_list with privacy labels
        """
        for data in data_list:
            # If no sensitive attributes identified, use all features
            if not sensitive_attributes:
                sensitive_attributes = list(range(data.x.size(1)))
            
            # Calculate privacy risk score based on sensitive attributes
            # Higher score means higher privacy risk
            sensitive_features = data.x[:, sensitive_attributes]
            
            # Risk increases with number of sensitive attributes exposed
            exposure_score = sensitive_features.sum(dim=1) / max(len(sensitive_attributes), 1)
            
            # Risk increases with node degree (more connections = more exposure)
            degrees = torch.zeros(data.x.size(0))
            for i in range(data.edge_index.size(1)):
                degrees[data.edge_index[0, i]] += 1
            
            # Normalize degrees (avoid division by zero)
            max_degree = degrees.max()
            if max_degree > 0:
                norm_degrees = degrees / max_degree
            else:
                norm_degrees = torch.zeros_like(degrees)
            
            # Combine exposure and degree factors for final privacy risk score
            privacy_risk = 0.7 * exposure_score + 0.3 * norm_degrees
            
            # Normalize to [0,1] range (avoid division by zero)
            min_risk = privacy_risk.min()
            max_risk = privacy_risk.max()
            if max_risk > min_risk:
                privacy_risk = (privacy_risk - min_risk) / (max_risk - min_risk)
            else:
                # If all values are the same, set to 0.5
                privacy_risk = torch.ones_like(privacy_risk) * 0.5
            
            data.privacy_risk = privacy_risk
            
        return data_list 