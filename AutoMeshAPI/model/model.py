import torch
import os
from model.encoder.encoders import TransformerTextEncoder, GraphEncoder, FeatureFusion, VectorQuantization
from model.decoder.decoder import GraphDecoder

def load_model(state_path="trained_model.pth", device=None):
    """
    Loads the model components with their respective state dictionaries.
    
    Args:
        state_path (str): Path to the saved state dictionary.
        device (torch.device, optional): Device to load the model on.
            Defaults to CUDA if available, else CPU.
            
    Returns:
        dict: A dictionary containing the loaded model components.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reinstantiate each model component with the same architecture as during training
    model = {
        "text_encoder": TransformerTextEncoder(embed_dim=768, num_heads=4, num_layers=2).to(device),
        "graph_encoder": GraphEncoder(in_channels=3, hidden_channels=32, out_channels=64, num_layers=3).to(device),
        "feature_fusion": FeatureFusion(text_dim=768, graph_dim=64, fused_dim=128).to(device),
        "vector_quantizer": VectorQuantization(codebook_size=512, embedding_dim=128).to(device),
        "graph_decoder": GraphDecoder(latent_dim=128, node_dim=3).to(device),
    }

    # Load the saved state dictionaries
    saved_state = torch.load(state_path, map_location=device)
    for name, part in model.items():
        part.load_state_dict(saved_state[name])
    
    return model

# Example usage:
# import torch
from model.encoder.encoders import TransformerTextEncoder, GraphEncoder, FeatureFusion, VectorQuantization
from model.decoder.decoder import GraphDecoder

def load_model(device=None):
    """
    Loads the model components with their respective state dictionaries.
    
    Args:
        state_path (str): Path to the saved state dictionary.
        device (torch.device, optional): Device to load the model on.
            Defaults to CUDA if available, else CPU.
            
    Returns:
        dict: A dictionary containing the loaded model components.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reinstantiate each model component with the same architecture as during training
    model = {
        "text_encoder": TransformerTextEncoder(embed_dim=768, num_heads=4, num_layers=2).to(device),
        "graph_encoder": GraphEncoder(in_channels=3, hidden_channels=32, out_channels=64, num_layers=3).to(device),
        "feature_fusion": FeatureFusion(text_dim=768, graph_dim=64, fused_dim=128).to(device),
        "vector_quantizer": VectorQuantization(codebook_size=512, embedding_dim=128).to(device),
        "graph_decoder": GraphDecoder(latent_dim=128, node_dim=3).to(device),
    }
    dir_path = os.path.dirname(os.path.realpath(__file__))
    state_path = os.path.join(dir_path, "trained_model.pth")
    # Load the saved state dictionaries
    saved_state = torch.load(state_path, map_location=device)
    for name, part in model.items():
        part.load_state_dict(saved_state[name])
    
    return model

# Example usage:
# model = load_model()

