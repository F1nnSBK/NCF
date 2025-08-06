import numpy as np
import torch
from torch.utils.data import Dataset


def get_user_item_maps(interactions_df):
    user_id_map = {id: idx for idx, id in enumerate(interactions_df['user_pseudo_id'].unique())}
    item_id_map = {id: idx for idx, id in enumerate(interactions_df['article_id'].unique())}
    return user_id_map, item_id_map

class InteractionsDataset(Dataset):
    def __init__(self, interactions_df):
        # Get mappings from user and item IDs to continuous integer indices
        self.user_id_map, self.item_id_map = get_user_item_maps(interactions_df)

        # Convert the data to tensors
        self.users = torch.tensor([self.user_id_map[user] for user in interactions_df["user_pseudo_id"]], dtype=torch.long)
        self.items = torch.tensor([self.item_id_map[item] for item in interactions_df["article_id"]], dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]
