import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from model import PrivacyGNN, PrivacyGNNWithExplanation

class MultiViewPrivacyGNN(torch.nn.Module):
    """
    Multi-view GNN model for privacy risk assessment that combines Facebook and Twitter data.
    Uses separate GNN models for each platform and combines their outputs.
    """
    def __init__(self, fb_input_dim, tw_input_dim, fb_feature_names, tw_feature_names, 
                 hidden_dim=64, output_dim=1, num_layers=3, dropout=0.2, conv_type='GAT'):
        super(MultiViewPrivacyGNN, self).__init__()
        
        # Store dimensions for later use
        self.hidden_dim = hidden_dim
        
        # Create separate models for Facebook and Twitter
        self.fb_model = PrivacyGNNWithExplanation(
            input_dim=fb_input_dim,
            feature_names=fb_feature_names,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output embeddings instead of scores
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        
        self.tw_model = PrivacyGNNWithExplanation(
            input_dim=tw_input_dim,
            feature_names=tw_feature_names,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output embeddings instead of scores
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        )
        
        # Feature attention for cross-platform integration
        # Determine the actual input dimension from the embeddings
        actual_input_dim = self.get_embedding_dim() * 2
        
        self.platform_attention = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final output layer
        output_input_dim = self.get_embedding_dim()
        self.output_layer = nn.Sequential(
            nn.Linear(output_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def get_embedding_dim(self):
        """Get the actual embedding dimension from the model."""
        # For models with structural features, the embedding dimension is larger
        if hasattr(self.fb_model, 'use_structural_features') and self.fb_model.use_structural_features:
            # The exact dimension depends on how structural features are incorporated
            # This is an estimate based on the original model
            return self.hidden_dim * 2
        return self.hidden_dim
    
    def forward_single_platform(self, x, edge_index, platform):
        """Forward pass for a single platform."""
        if platform == "facebook":
            # Get embeddings from Facebook model (without final sigmoid)
            h = self.fb_model.forward_embeddings(x, edge_index)
        else:  # twitter
            # Get embeddings from Twitter model (without final sigmoid)
            h = self.tw_model.forward_embeddings(x, edge_index)
        
        return h
    
    def forward_dual_platform(self, fb_data, tw_data):
        """Forward pass with data from both platforms."""
        # Get embeddings from both platforms
        fb_embeddings = self.fb_model.forward_embeddings(fb_data.x, fb_data.edge_index)
        tw_embeddings = self.tw_model.forward_embeddings(tw_data.x, tw_data.edge_index)
        
        # For each node in Facebook, find the most similar node in Twitter
        # based on cosine similarity of embeddings
        combined_embeddings = []
        
        for fb_emb in fb_embeddings:
            # Compute similarities with all Twitter embeddings
            similarities = F.cosine_similarity(
                fb_emb.unsqueeze(0).expand(tw_embeddings.size(0), -1),
                tw_embeddings,
                dim=1
            )
            
            # Get the most similar Twitter embedding
            _, max_idx = similarities.max(dim=0)
            tw_emb = tw_embeddings[max_idx]
            
            # Concatenate embeddings
            concat_emb = torch.cat([fb_emb, tw_emb], dim=0)
            
            # Apply attention to weight the importance of each platform
            # Reshape concat_emb to have batch dimension
            concat_emb_reshaped = concat_emb.view(1, -1)
            attention = self.platform_attention(concat_emb_reshaped).squeeze(0)
            
            # Apply attention weights to get weighted embedding
            fb_dim = fb_emb.size(0)
            weighted_emb = torch.cat([
                attention[0] * fb_emb,
                attention[1] * tw_emb[:fb_dim]  # Ensure same dimension as fb_emb
            ])
            
            # Alternative: simple average if dimensions match
            if fb_emb.size(0) == tw_emb.size(0):
                weighted_emb = attention[0] * fb_emb + attention[1] * tw_emb
            
            combined_embeddings.append(weighted_emb)
        
        combined_embeddings = torch.stack(combined_embeddings)
        
        # Apply final output layer
        privacy_risk = self.output_layer(combined_embeddings)
        
        # Normalize scores to [0, 1] range
        privacy_risk = torch.sigmoid(privacy_risk)
        
        return privacy_risk
    
    def forward(self, x=None, edge_index=None, platform=None, fb_data=None, tw_data=None):
        """
        Forward pass with flexible input options.
        
        Args:
            x: Node features (for single platform)
            edge_index: Edge indices (for single platform)
            platform: Platform identifier ("facebook" or "twitter") for single platform
            fb_data: Facebook data object (for dual platform)
            tw_data: Twitter data object (for dual platform)
        """
        if fb_data is not None and tw_data is not None:
            # Dual platform mode
            return self.forward_dual_platform(fb_data, tw_data)
        elif x is not None and edge_index is not None and platform is not None:
            # Single platform mode
            embeddings = self.forward_single_platform(x, edge_index, platform)
            privacy_risk = self.output_layer(embeddings)
            return torch.sigmoid(privacy_risk)
        else:
            raise ValueError("Invalid input combination. Provide either (x, edge_index, platform) or (fb_data, tw_data).")
    
    def loss(self, pred, target):
        """Calculate loss between predicted and target privacy risk scores."""
        # Ensure no NaN values
        valid_mask = ~torch.isnan(pred) & ~torch.isnan(target)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # MSE loss for regression
        loss = F.mse_loss(pred[valid_mask], target[valid_mask])
        return loss
    
    def explain_prediction(self, fb_data, tw_data):
        """
        Explain privacy risk prediction by identifying important features from both platforms.
        
        Args:
            fb_data: Facebook data object
            tw_data: Twitter data object
            
        Returns:
            Dict with feature importance scores and explanation
        """
        # Get explanations from individual models
        fb_explanation = self.fb_model.explain_prediction(fb_data.x, fb_data.edge_index)
        tw_explanation = self.tw_model.explain_prediction(tw_data.x, tw_data.edge_index)
        
        # Combine explanations
        combined_explanation = {
            "facebook": fb_explanation,
            "twitter": tw_explanation,
        }
        
        # Get top features from both platforms
        fb_features = fb_explanation["important_features"]
        tw_features = tw_explanation["important_features"]
        
        # Create combined list of important features
        important_features = []
        for i in range(min(3, len(fb_features))):
            important_features.append({
                "platform": "facebook",
                "feature": fb_features[i]["feature"],
                "importance": fb_features[i]["importance"]
            })
        
        for i in range(min(3, len(tw_features))):
            important_features.append({
                "platform": "twitter",
                "feature": tw_features[i]["feature"],
                "importance": tw_features[i]["importance"]
            })
        
        # Sort by importance
        important_features = sorted(important_features, key=lambda x: x["importance"], reverse=True)
        
        combined_explanation["combined_important_features"] = important_features
        
        return combined_explanation

# Add forward_embeddings method to PrivacyGNNWithExplanation
def forward_embeddings(self, x, edge_index, batch=None):
    """Forward pass that returns embeddings instead of privacy risk scores."""
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
    
    return h

# Add the method to the PrivacyGNNWithExplanation class
PrivacyGNNWithExplanation.forward_embeddings = forward_embeddings 