import torch
import torch.nn as nn

class NodeCountPredictor(nn.Module):
    def __init__(self, latent_dim):
        super(NodeCountPredictor, self).__init__()
        self.node_count_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Predict a single scalar
        )

    def forward(self, z):
        """
        Args:
            z: Latent representation [batch_size, latent_dim]
        Returns:
            predicted_node_count: Scalar prediction for the number of nodes
        """
        return self.node_count_predictor(z).squeeze(-1)

def pretrain_node_count_predictor(predictor, data_loader, optimizer, device, num_epochs=10):
    predictor.train()
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            z = data.latent_vector  # Use precomputed latent vector
            true_num_nodes = data.num_nodes  # Ground truth node count
            
            # Forward pass
            predicted_num_nodes = predictor(z)
            
            # Compute loss
            loss = loss_fn(predicted_num_nodes, true_num_nodes.float())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader):.4f}")


class EdgeDecoder(nn.Module):
    def __init__(self, node_dim, hidden_dim=64):
        super(EdgeDecoder, self).__init__()
        # This MLP takes the concatenated pair of node embeddings and outputs a scalar probability.
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings):
        """
        Args:
            node_embeddings: Tensor of shape [num_nodes, node_dim]
        Returns:
            adj: Tensor of shape [num_nodes, num_nodes] with edge probabilities.
        """
        num_nodes = node_embeddings.size(0)
        # Expand node embeddings to construct pairwise combinations:
        # For each pair (i, j), we concatenate the embeddings.
        # Efficient method: use broadcasting.
        # node_embeddings_i: [num_nodes, 1, node_dim]
        # node_embeddings_j: [1, num_nodes, node_dim]
        node_embeddings_i = node_embeddings.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        node_embeddings_j = node_embeddings.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        pairwise_input = torch.cat([node_embeddings_i, node_embeddings_j], dim=-1)  # shape: [num_nodes, num_nodes, 2 * node_dim]
        
        # Apply MLP to each pair
        # Reshape for MLP: [num_nodes*num_nodes, 2 * node_dim]
        pairwise_input_flat = pairwise_input.view(-1, pairwise_input.shape[-1])
        edge_logits = self.mlp(pairwise_input_flat)
        edge_logits = edge_logits.view(num_nodes, num_nodes)
        
        # Optionally enforce symmetry by averaging with the transpose:
        edge_logits = (edge_logits + edge_logits.T) / 2.0
        
        # Convert logits to probabilities with a sigmoid
        adj = torch.sigmoid(edge_logits)
        return adj

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, node_dim, edge_hidden_dim=64):
        super(GraphDecoder, self).__init__()
        self.node_decoder = nn.Linear(latent_dim, node_dim)
        self.node_count_predictor = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.ReLU()  # Ensure non-negative outputs
        )
        self.node_count_predictor[0].weight.data.uniform_(0.1, 1.0)
        self.node_count_predictor[0].bias.data.fill_(10)  # Initialize bias to a higher value
        
        # Edge decoder to predict the adjacency matrix
        self.edge_decoder = EdgeDecoder(node_dim, hidden_dim=edge_hidden_dim)

    def forward(self, z, true_num_nodes:int=1):
        """
        Args:
            z: Latent representation [batch_size, latent_dim]
        Returns:
            reconstructed_nodes: Tensor of shape [num_predicted_nodes, node_dim]
            reconstructed_adj: Tensor of shape [num_predicted_nodes, num_predicted_nodes]
        """
        batch_size = z.size(0)
        assert batch_size == 1, "Currently, this decoder supports only batch_size=1."

        # Predict the number of nodes
        raw_node_counts = self.node_count_predictor(z).squeeze(-1)  # Shape: [batch_size]
        predicted_num_nodes = torch.clamp(raw_node_counts.round(), min=50, max=10000).long().item()

        # Expand latent vector for node decoding
        if true_num_nodes == 1:
            z_expanded = z.expand(predicted_num_nodes, -1)  # [predicted_num_nodes, latent_dim]
        else:
            z_expanded = z.expand(true_num_nodes, -1)

        # Decode nodes
        reconstructed_nodes = self.node_decoder(z_expanded)  # [predicted_num_nodes, node_dim]

        # Generate adjacency matrix with the new edge decoder
        reconstructed_adj = self.edge_decoder(reconstructed_nodes)  # [predicted_num_nodes, predicted_num_nodes]

        return reconstructed_nodes, reconstructed_adj
