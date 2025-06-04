import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class PrivacyGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.2, 
                 conv_type='GCN', use_node_features=True, use_structural_features=True):
        """
        Privacy Risk Assessment GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension size
            output_dim: Output dimension (1 for privacy risk score)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            conv_type: Type of GNN layer (GCN, GAT, SAGE)
            use_node_features: Whether to use node features
            use_structural_features: Whether to use structural features (degree, clustering, etc.)
        """
        super(PrivacyGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_node_features = use_node_features
        self.use_structural_features = use_structural_features
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer takes input_dim -> hidden_dim
        if conv_type == 'GCN':
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        elif conv_type == 'SAGE':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Rest of the layers
        for _ in range(num_layers - 1):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Structural feature layers (degree, clustering coefficient, etc.)
        if self.use_structural_features:
            self.struct_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            final_dim = hidden_dim * 2  # Combined with GNN output
        else:
            final_dim = hidden_dim
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def compute_structural_features(self, edge_index, num_nodes):
        """Compute structural features for each node (degree, clustering, etc.)"""
        device = edge_index.device
        
        # Compute node degrees
        degrees = torch.zeros(num_nodes, device=device)
        for i in range(edge_index.size(1)):
            degrees[edge_index[0, i]] += 1
        
        # Normalize degrees (avoid division by zero)
        max_degree = degrees.max()
        if max_degree > 0:
            norm_degrees = degrees / max_degree
        else:
            norm_degrees = degrees
        
        # Compute clustering coefficients (approximation)
        # For each node, compute ratio of neighbors that are connected to each other
        clustering = torch.zeros(num_nodes, device=device)
        
        # Compute local influence score (eigenvector centrality approximation)
        # Simple approximation: average of neighbors' degrees
        influence = torch.zeros(num_nodes, device=device)
        for i in range(num_nodes):
            neighbors = torch.where(edge_index[0, :] == i)[0]
            neighbor_nodes = edge_index[1, neighbors]
            if len(neighbor_nodes) > 0:
                influence[i] = torch.mean(degrees[neighbor_nodes])
        
        # Normalize influence (avoid division by zero)
        max_influence = influence.max()
        if max_influence > 0:
            influence = influence / max_influence
        
        # Combine structural features
        structural_features = torch.stack([norm_degrees, clustering, influence], dim=1)
        return structural_features
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        # Project input features
        if self.use_node_features:
            h = self.input_proj(x)
        else:
            # If not using node features, use random features
            h = torch.randn(x.size(0), self.hidden_dim, device=x.device)
        
        # Apply GNN layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Compute and incorporate structural features
        if self.use_structural_features:
            structural_features = self.compute_structural_features(edge_index, x.size(0))
            structural_embedding = self.struct_encoder(structural_features)
            
            # Combine GNN output with structural features
            h = torch.cat([h, structural_embedding], dim=1)
        
        # Apply output layers to get privacy risk scores
        privacy_risk = self.output_layers(h)
        
        # Normalize scores to [0, 1] range
        privacy_risk = torch.sigmoid(privacy_risk)
        
        return privacy_risk
    
    def loss(self, pred, target):
        """Calculate loss between predicted and target privacy risk scores."""
        # Ensure no NaN values
        valid_mask = ~torch.isnan(pred) & ~torch.isnan(target)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # MSE loss for regression
        loss = F.mse_loss(pred[valid_mask], target[valid_mask])
        return loss

class PrivacyGNNWithExplanation(PrivacyGNN):
    """Extension of PrivacyGNN with attention-based feature attribution for explainability."""
    
    def __init__(self, input_dim, feature_names, hidden_dim=64, output_dim=1, num_layers=3, 
                 dropout=0.2, conv_type='GAT', use_node_features=True, use_structural_features=True):
        super(PrivacyGNNWithExplanation, self).__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
            num_layers=num_layers, dropout=dropout, conv_type=conv_type,
            use_node_features=use_node_features, use_structural_features=use_structural_features
        )
        self.feature_names = feature_names
        
        # Feature attention layer
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None, return_attention=False):
        """Forward pass with feature attribution."""
        # Calculate feature attention weights
        feature_weights = self.feature_attention(x)
        
        # Apply feature weights to input
        weighted_x = x * feature_weights
        
        # Use the parent class's forward method with weighted features
        if self.use_node_features:
            h = self.input_proj(weighted_x)
        else:
            h = torch.randn(x.size(0), self.hidden_dim, device=x.device)
        
        # Apply GNN layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Compute and incorporate structural features
        if self.use_structural_features:
            structural_features = self.compute_structural_features(edge_index, x.size(0))
            structural_embedding = self.struct_encoder(structural_features)
            
            # Combine GNN output with structural features
            h = torch.cat([h, structural_embedding], dim=1)
        
        # Apply output layers to get privacy risk scores
        privacy_risk = self.output_layers(h)
        
        # Normalize scores to [0, 1] range
        privacy_risk = torch.sigmoid(privacy_risk)
        
        if return_attention:
            # Return both the risk scores and the feature attention weights
            return privacy_risk, feature_weights
        
        return privacy_risk
    
    def explain_prediction(self, x, edge_index, node_idx=None):
        """
        Explain model prediction by identifying most important features.
        
        Args:
            x: Node feature matrix
            edge_index: Edge indices
            node_idx: Index of node to explain (None for all nodes)
            
        Returns:
            Dict with feature importance scores and explanation
        """
        # Forward pass with attention
        _, feature_weights = self.forward(x, edge_index, return_attention=True)
        
        if node_idx is not None:
            # Get attention weights for specific node
            node_attention = feature_weights[node_idx]
        else:
            # Average attention across all nodes
            node_attention = torch.mean(feature_weights, dim=0)
        
        # Get top 5 most important features
        top_values, top_indices = torch.topk(node_attention, min(5, len(self.feature_names)))
        
        # Create explanation
        important_features = []
        for i, idx in enumerate(top_indices):
            feature_name = self.feature_names[idx]
            importance = top_values[i].item()
            important_features.append({
                "feature": feature_name,
                "importance": importance
            })
        
        explanation = {
            "important_features": important_features,
            "all_feature_weights": node_attention.detach().cpu().numpy(),
            "feature_names": self.feature_names
        }
        
        return explanation 