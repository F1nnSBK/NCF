import numpy as np
import optuna
import polars as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ncf_training.trainer.utils.ncf import NCF
from utils.interactions_dataset import InteractionsDataset

# --- 1. Datenvorbereitung (Einmalig) ---
data = pl.read_csv("data/ncf_data_v1.csv")
data = data.drop_nulls()
data = data.sample(n=1_000_000, seed=42)
print(data["user_pseudo_id"].n_unique(), "unique users")
print(data["article_id"].n_unique(), "unique items")
print(data.shape, "rows in the dataset")

train_df, test_df = train_test_split(data.to_pandas(), test_size=0.2, random_state=42)

train_df = pl.from_pandas(train_df)
test_df = pl.from_pandas(test_df)

print("Training DataFrame:")
print(train_df.shape)
print("\nTest DataFrame:")
print(test_df.shape)

train_dataset = InteractionsDataset(train_df)
test_dataset = InteractionsDataset(test_df)

num_users = len(train_dataset.user_id_map)
num_items = len(train_dataset.item_id_map)
print(f"Number of users: {num_users}\nNumber of items: {num_items}")

# --- NDCG und DCG Funktionen ---
def dcg_at_k(scores, k=10):
    if len(scores) == 0:
        return 0.0
    scores = np.array(scores[:k])
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(truth, scores, k=10):
    best_dcg = dcg_at_k(sorted(truth, reverse=True), k)
    actual_dcg = dcg_at_k([truth[i] for i in np.argsort(scores)[::-1]], k)
    return actual_dcg / best_dcg if best_dcg > 0 else 0

# Datenmapping für die Evaluierung vorbereiten (einmalig)
user_id_map = train_dataset.user_id_map
item_id_map = train_dataset.item_id_map
train_df = train_df.with_columns(
    pl.col("user_pseudo_id").map_elements(lambda x: user_id_map.get(x), return_dtype=pl.Int64).alias("user_id_map"),
    pl.col("article_id").map_elements(lambda x: item_id_map.get(x), return_dtype=pl.Int64).alias("item_id_map")
)
test_df = test_df.with_columns(
    pl.col("user_pseudo_id").map_elements(lambda x: user_id_map.get(x), return_dtype=pl.Int64).alias("user_id_map"),
    pl.col("article_id").map_elements(lambda x: item_id_map.get(x), return_dtype=pl.Int64).alias("item_id_map")
)
test_users = test_df["user_id_map"].unique()
user_history_df = train_df.group_by("user_id_map").agg(
    pl.col("item_id_map").alias("item_id_map_list")
)
user_history = dict(zip(user_history_df["user_id_map"], user_history_df["item_id_map_list"]))


# --- 2. Objective-Funktion für Optuna ---
def objective(trial):
    # Hyperparameter vorschlagen
    embedding_dim = trial.suggest_int('embedding_dim', 16, 128, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
    num_epochs = trial.suggest_int('epochs', 10, 30)

    # DataLoader mit vorgeschlagener Batch-Size neu erstellen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modell initialisieren
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users, num_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Trainings-Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user_idxs, item_idxs in train_loader:
            user_idxs_pos, item_idxs_pos = user_idxs.to(device), item_idxs.to(device)
            labels_pos = torch.ones(len(user_idxs_pos), device=device)

            neg_items = torch.randint(0, num_items, size=(len(user_idxs) * 4,), device=device)
            user_idxs_neg = user_idxs.repeat_interleave(4).to(device)
            item_idxs_neg = neg_items.to(device)
            labels_neg = torch.zeros(len(user_idxs_neg), device=device)

            combined_user_idxs = torch.cat([user_idxs_pos, user_idxs_neg], dim=0)
            combined_item_idxs = torch.cat([item_idxs_pos, item_idxs_neg], dim=0)
            combined_labels = torch.cat([labels_pos, labels_neg], dim=0)

            optimizer.zero_grad()
            outputs = model(combined_user_idxs, combined_item_idxs)
            loss = criterion(outputs.view(-1), combined_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluierungs-Loop
    model.eval()
    ndcg_scores = []
    with torch.no_grad():
        for user_id in test_users:
            if user_id is None:
                continue
            pos_items = test_df.filter(pl.col("user_id_map") == user_id)["item_id_map"].to_list()
            pos_items = [item for item in pos_items if item is not None]

            if not pos_items:
                continue

            user_interacted_items = set(user_history.get(user_id, []))
            neg_items = []
            while len(neg_items) < 99:
                neg_item = np.random.randint(0, num_items - 1)
                if neg_item not in user_interacted_items and neg_item not in pos_items and neg_item not in neg_items:
                    neg_items.append(neg_item)

            all_items = pos_items + neg_items
            true_scores = np.zeros(len(all_items))
            true_scores[:len(pos_items)] = 1
            user_tensor = torch.LongTensor([user_id] * len(all_items)).to(device)
            item_tensor = torch.LongTensor(all_items).to(device)
            scores = model(user_tensor, item_tensor).cpu().numpy().flatten()
            ndcg_scores.append(ndcg_at_k(true_scores, scores, k=10))

    average_ndcg = np.mean(ndcg_scores)
    return average_ndcg


# --- 3. Optuna-Studie starten ---
if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize', 
        storage='sqlite:///hyper.db',
        study_name='ncf_hpo_january_small'
    )
    study.optimize(objective, n_trials=50)

    print("Beste Trial:")
    print(study.best_trial.params)
    print("Bester nDCG@10 Wert:")
    print(study.best_value)