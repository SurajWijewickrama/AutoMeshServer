import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from get_data_point import load_single_data_point
from fastapi import HTTPException
from model.model import load_model
def generate_graph(data, device):
    """
    Runs inference on a single data point using the scripted model.
    
    Args:
        data (torch_geometric.data.Data): The input data point.
        device (torch.device): Device to run inference on.
        
    Returns:
        (reconstructed_nodes, reconstructed_adj): The decoder outputs.
    """
    # Load the  model (assumes the model has been exported with TorchScript)
    model = load_model()
    
    print(model)
    
    # Move the data point to the device
    data = data.to(device)
    text_input = data.text_label      # shape: [batch_size, seq_len, embed_dim]
    graph_x = data.x                  # shape: [num_nodes, feature_dim]
    edge_index = data.edge_index      # shape: [2, num_edges]
    
    # Use the number of nodes from the input data
    c_text = model['text_encoder'](text_input.unsqueeze(1))  # [batch_size, seq_len, embed_dim]
    z_G = model['graph_encoder'](graph_x, edge_index)          # [graph_dim] or [batch_size, graph_dim]
    
    # Fuse features and quantize
    fused_features = model['feature_fusion'](c_text, z_G)       # [batch_size, fused_dim]
    z_qG, _ = model['vector_quantizer'](fused_features)         # [batch_size, fused_dim]

    # Decode graph
    # Here we assume batch_size is 1.
    reconstructed_nodes, reconstructed_adj = model['graph_decoder'](z_qG, graph_x.size(0))
    
    return reconstructed_nodes, reconstructed_adj

def generate_output_json(prompt,
                         location=[0.0, 0.0, 0.0],
                         rotation=[0.0, 0.0, 0.0],
                         scale=[1.0, 1.0, 1.0],
                         edge_threshold=0.1):
    """
    Loads a sample data point, runs it through the generation process,
    and produces a JSON-serializable dictionary with the reconstructed graph.
    
    The expected output format is:
    
    {
      "n": "prompt",
      "l": [0.0, 0.0, 0.0],
      "r": [0.0, 0.0, 0.0],
      "s": [1.0, 1.0, 1.0],
      "v": [ [x1, y1, z1], [x2, y2, z2], ... ],    # vertices
      "e": [ [i1, j1], [i2, j2], ... ]              # edges
    }
    """
    # Load a sample data point from a JSON file.
    file_path = os.path.join(os.getcwd(), "sample_data.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Data file not found")
        
    with open(file_path, 'r') as file:
        data_json = json.load(file)
    
    # Convert JSON data into a PyG Data object.
    # This example assumes the JSON has the fields:
    #   "v": vertices,
    #   "e": edge indices,
    #   "text_label": pre-computed text features.
    data_point = load_single_data_point("datapoint")  # shape: [1, embed_dim]
    
    print(data_point)
    
    # Run the generation process on the data point
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstructed_nodes, reconstructed_adj = generate_graph(data_point, device)
    
    # Convert node features: take the first 3 dimensions as (x, y, z) coordinates.
    nodes_np = reconstructed_nodes[:, :3].cpu().detach().numpy()
    nodes_list = nodes_np.tolist()
    
    # Process the reconstructed adjacency matrix:
    # Convert it to a NumPy array, threshold it to get binary edges,
    # then extract edge indices from the upper triangle.
    adj_np = reconstructed_adj.cpu().detach().numpy()
    binary_adj = (adj_np > edge_threshold).astype(int)
    # Extract edges (upper triangle to avoid duplicates and self-loops)
    i_indices, j_indices = np.where(np.triu(binary_adj, k=1))
    edges_list = [[int(i), int(j)] for i, j in zip(i_indices, j_indices)]
    
    # Construct the final JSON object in the desired format.
    output = {
        "n": prompt,
        "l": location,
        "r": rotation,
        "s": scale,
        "v": nodes_list,
        "e": edges_list,
        "f": []
    }
    return output
