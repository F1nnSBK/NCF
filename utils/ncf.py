import torch
import torch.functional as F
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[64, 32, 16]):
        super(NCF, self).__init__()
        # GMF part
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        for layer_size in layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp =  nn.Sequential(*mlp_layers)

        # Fusion layer
        self.final_layer = nn.Linear(embedding_dim + layers[-1], 1)

    def forward(self, user_indices, item_indices):
        # GMF forward pass
        gmf_user_latent = self.user_embedding_gmf(user_indices)
        gmf_item_latent = self.item_embedding_gmf(item_indices)
        gmf_output = gmf_user_latent * gmf_item_latent

        # MLP forward pass
        mlp_user_latent = self.user_embedding_mlp(user_indices)
        mlp_item_latent = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Fusion
        final_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.final_layer(final_input))

        return prediction