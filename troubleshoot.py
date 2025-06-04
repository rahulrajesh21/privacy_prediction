import os
import sys
import torch
import pandas as pd
import numpy as np

print("=== Troubleshooting Facebook Data Analysis ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Current working directory: {os.getcwd()}")

# Check if the facebook directory exists
print("\nChecking for facebook directory...")
if not os.path.exists("facebook"):
    print("ERROR: 'facebook' directory not found in current working directory")
    print("Please make sure you're running the script from the correct directory")
    sys.exit(1)
else:
    print("Found 'facebook' directory")

# Check for .edges files in the facebook directory
print("\nChecking for ego network files...")
edge_files = [f for f in os.listdir("facebook") if f.endswith(".edges")]
if not edge_files:
    print("ERROR: No .edges files found in facebook directory")
    sys.exit(1)
else:
    print(f"Found {len(edge_files)} .edges files: {', '.join(edge_files[:5])}...")

# Try to load one of the edge files
print("\nTrying to load the first edge file...")
try:
    first_edge_file = edge_files[0]
    ego_id = first_edge_file.split(".")[0]
    print(f"Testing with ego_id: {ego_id}")
    
    edges_file_path = os.path.join("facebook", f"{ego_id}.edges")
    feat_file_path = os.path.join("facebook", f"{ego_id}.feat")
    featnames_file_path = os.path.join("facebook", f"{ego_id}.featnames")
    
    # Check if all required files exist
    all_files_exist = os.path.exists(edges_file_path) and os.path.exists(feat_file_path) and os.path.exists(featnames_file_path)
    
    if not all_files_exist:
        print("ERROR: Not all required files exist for ego_id:", ego_id)
        print(f"  .edges file exists: {os.path.exists(edges_file_path)}")
        print(f"  .feat file exists: {os.path.exists(feat_file_path)}")
        print(f"  .featnames file exists: {os.path.exists(featnames_file_path)}")
        sys.exit(1)
    
    # Try to load the edges file
    try:
        edges_df = pd.read_csv(edges_file_path, sep=' ', header=None, names=['source', 'target'])
        print(f"Successfully loaded edges file with {len(edges_df)} edges")
    except Exception as e:
        print(f"ERROR loading edges file: {str(e)}")
        sys.exit(1)
    
    # Try to load the features file
    try:
        feat_df = pd.read_csv(feat_file_path, sep=' ', header=None)
        print(f"Successfully loaded features file with {feat_df.shape[0]} nodes and {feat_df.shape[1]-1} features")
    except Exception as e:
        print(f"ERROR loading features file: {str(e)}")
        sys.exit(1)
    
    # Try to load the feature names file
    try:
        feature_names = []
        with open(featnames_file_path, 'r') as f:
            for line in f:
                if ' ' in line.strip():  # Make sure we can split the line
                    idx, name = line.strip().split(' ', 1)
                    feature_names.append(name)
        print(f"Successfully loaded feature names file with {len(feature_names)} feature names")
    except Exception as e:
        print(f"ERROR loading feature names file: {str(e)}")
        sys.exit(1)
    
    # Try to create PyTorch tensors
    try:
        # Convert node IDs and edges to proper format
        node_ids = feat_df[0].values
        features = feat_df.iloc[:, 1:].values
        
        # Create node ID mapping
        node_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
        
        # Map edge IDs
        edge_index = []
        for _, row in edges_df.iterrows():
            if row['source'] in node_map and row['target'] in node_map:
                edge_index.append([node_map[row['source']], node_map[row['target']]])
                # Add the reverse edge for undirected graph
                edge_index.append([node_map[row['target']], node_map[row['source']]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        x = torch.tensor(features, dtype=torch.float)
        
        print(f"Successfully created PyTorch tensors:")
        print(f"  Node features tensor shape: {x.shape}")
        print(f"  Edge index tensor shape: {edge_index.shape}")
        
    except Exception as e:
        print(f"ERROR creating PyTorch tensors: {str(e)}")
        sys.exit(1)
    
except Exception as e:
    print(f"Unexpected error during testing: {str(e)}")
    sys.exit(1)

print("\nAll basic tests passed! The data can be loaded successfully.")
print("\nNext steps to troubleshoot your issue:")
print("1. Check if torch-geometric is properly installed:")
print("   Run: pip list | grep torch")
print("\n2. If you're using CUDA, check your CUDA version:")
print("   Run: nvidia-smi or nvcc --version")
print("\n3. Try to install torch-geometric with the specific version:")
print("   pip install torch-geometric")
print("   pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html")
print("   Replace {TORCH_VERSION} with your PyTorch version and {CUDA_VERSION} with your CUDA version or 'cpu'")
print("\n4. If you're still having issues, try running with CPU only by setting:")
print("   export CUDA_VISIBLE_DEVICES=''")

print("\nIf everything looks good with the data, the issue might be with the GCN model.")
print("Try running a simplified example to test:")
print("python example_usage.py") 