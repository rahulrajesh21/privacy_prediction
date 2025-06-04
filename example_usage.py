import torch
import numpy as np
import matplotlib.pyplot as plt
from gnn_privacy_predictor import (
    load_facebook_data, 
    GCNPrivacyPredictor, 
    calculate_privacy_score,
    simulate_privacy_enhancement
)

def main():
    """
    Demonstrate how to use the GCNPrivacyPredictor model on a specific ego network.
    """
    print("AI-Powered Privacy Risk Prediction for Social Media Users")
    print("=" * 60)
    
    # 1. Load a specific ego network
    ego_id = "0"  # Change this to use a different ego network
    
    # Optionally specify a specific sensitive attribute index
    # For example, to use "education;school;id;anonymized feature 24" as sensitive attribute:
    # sensitive_attr_idx = 24
    sensitive_attr_idx = None  # Will be randomly selected if None
    
    print(f"Loading ego network {ego_id}...")
    data, sensitive_attr_idx, feature_names = load_facebook_data(ego_id, sensitive_attr_idx)
    
    print(f"Loaded network with {data.x.shape[0]} nodes, {data.edge_index.shape[1]//2} edges")
    print(f"Selected sensitive attribute: {feature_names[sensitive_attr_idx]}")
    
    # 2. Calculate initial privacy scores for each user
    print("\nCalculating initial privacy scores...")
    initial_privacy_scores = []
    for node_idx in range(len(data.x)):
        # Find neighbors of this node
        neighbors = []
        for i in range(data.edge_index.shape[1]):
            if data.edge_index[0, i].item() == node_idx:
                neighbors.append(data.edge_index[1, i].item())
        
        # Get features of neighbors
        neighbors_features = [data.x[n] for n in neighbors]
        
        # Calculate privacy score
        score = calculate_privacy_score(data.x[node_idx], sensitive_attr_idx, neighbors_features)
        initial_privacy_scores.append(score)
    
    privacy_scores = torch.tensor(initial_privacy_scores, dtype=torch.float)
    
    # 3. Create and load a pre-trained model (for demonstration, we'll create a new one)
    print("\nCreating a GCN privacy predictor model...")
    model = GCNPrivacyPredictor(num_features=data.x.shape[1])
    
    # Here you would normally load pre-trained weights:
    # model.load_state_dict(torch.load('pretrained_model.pt'))
    
    # For demonstration, we'll just make random predictions
    # In a real scenario, you would use the trained model
    print("Making privacy risk predictions (simulated)...")
    # Random predictions between 0 and 1
    predicted_scores = torch.rand(data.x.shape[0])
    
    # 4. Analyze privacy risk distribution
    high_risk_count = (predicted_scores < 0.3).sum().item()
    medium_risk_count = ((predicted_scores >= 0.3) & (predicted_scores < 0.7)).sum().item()
    low_risk_count = (predicted_scores >= 0.7).sum().item()
    
    print(f"\nPrivacy Risk Analysis:")
    print(f"High Risk Users:    {high_risk_count} ({high_risk_count/len(predicted_scores)*100:.1f}%)")
    print(f"Medium Risk Users:  {medium_risk_count} ({medium_risk_count/len(predicted_scores)*100:.1f}%)")
    print(f"Low Risk Users:     {low_risk_count} ({low_risk_count/len(predicted_scores)*100:.1f}%)")
    
    # 5. Simulate privacy enhancement
    print("\nSimulating privacy enhancement...")
    enhanced_data, new_privacy_scores, utility_loss = simulate_privacy_enhancement(
        data, predicted_scores, sensitive_attr_idx, 
        privacy_threshold=0.3, intervention_rate=0.6
    )
    
    print(f"Privacy enhancement applied to high-risk users.")
    print(f"Utility loss: {utility_loss:.4f} ({utility_loss*100:.1f}% of users had their sensitive attribute generalized)")
    
    # 6. Visualize privacy scores before and after enhancement
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(predicted_scores.numpy(), bins=20, alpha=0.7, color='red')
    plt.title('Predicted Privacy Scores\nBefore Enhancement')
    plt.xlabel('Privacy Score (lower = higher risk)')
    plt.ylabel('Number of Users')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(new_privacy_scores.numpy(), bins=20, alpha=0.7, color='green')
    plt.title('Privacy Scores\nAfter Enhancement')
    plt.xlabel('Privacy Score (lower = higher risk)')
    plt.ylabel('Number of Users')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('privacy_enhancement_effect.png')
    
    print("\nExample completed! Check 'privacy_enhancement_effect.png' for visualization.")
    print("In a real application, you would use a fully trained model for more accurate predictions.")

if __name__ == "__main__":
    main() 