"""
champ_model.py
──────────────
Predicts the Drivers' Championship winner using a PyTorch Transformer.

The model takes a sequence of race results across a season (one vector per round)
and predicts whether that driver wins the championship.

Usage:
    python src/champ_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import (
    DATA_FEATURES, TRAIN_YEARS_CUTOFF,
    CHAMP_FEATURE_COLS, CHAMP_TARGET_COL,
    RANDOM_SEED, LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────
PREDICTION_ROUND = 5   # use only first 10 races
MAX_ROUNDS = PREDICTION_ROUND
#MAX_ROUNDS  = 25       # max races in a season (pad/truncate to this)
FEAT_DIM    = 8        # number of features per round
NHEAD       = 4        # transformer attention heads
NUM_LAYERS  = 2        # transformer encoder layers
DROPOUT     = 0.2
BATCH_SIZE  = 32
EPOCHS      = 40
LR          = 1e-3

# Features used per round (must exist in champ_features.parquet)
SEQUENCE_FEATURES = [
    "final_position",
    "points",
    "cumulative_wins",
    "races_remaining",
    "rolling_avg_finish",
    "rolling_dnfs",
    "rolling_points_last_3",
    "rolling_wins_last_5",
    "momentum_score",
    "consistency_score",
    "podium_rate",
    "win_rate",
    "reliability_score",
    "constructor_cumulative_points",
    "constructor_rolling_points",
    "constructor_momentum",
    "career_races",
]


# ── Spark Session ──────────────────────────────────────────────────────────

def get_spark():
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["HADOOP_HOME"]           = "C:\\Hadoop"

    spark = SparkSession.builder \
        .appName("F1ChampModel") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ── Data Loading ───────────────────────────────────────────────────────────

def load_champ_data(spark):
    log.info("Loading championship features...")

    path = os.path.join(DATA_FEATURES, "champ_features.parquet")
    df   = spark.read.parquet(path)

    # Only keep columns we need
    needed = ["driver_id", "driver_name", "year", "round",
              CHAMP_TARGET_COL] + SEQUENCE_FEATURES
    existing = [c for c in needed if c in df.columns]

    pdf = df.select(existing).toPandas()

    # Ensure numeric
    for col in SEQUENCE_FEATURES:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0)

    pdf[CHAMP_TARGET_COL] = pdf[CHAMP_TARGET_COL].fillna(0).astype(int)

    log.info(f"Loaded {len(pdf)} rows across {pdf['year'].nunique()} seasons")
    log.info(f"Champions in dataset: {pdf.groupby('year')[CHAMP_TARGET_COL].max().sum()}")
    return pdf


# ── Sequence Building ──────────────────────────────────────────────────────

def build_sequences(pdf):
    """
    Builds one sequence per (driver, year).

    IMPORTANT:
    Only the first PREDICTION_ROUND races are used.
    This converts the task into MID-SEASON championship prediction
    instead of full-season leakage prediction.

    Shape:
        (num_driver_seasons, MAX_ROUNDS, FEAT_DIM)

    Labels:
        1 if the driver eventually won the championship
    """

    log.info(f"Using first {PREDICTION_ROUND} races for championship forecasting")

    sequences = []
    labels    = []
    meta      = []

    available_feats = [f for f in SEQUENCE_FEATURES if f in pdf.columns]
    feat_dim = len(available_feats)

    # Group by driver-season
    for (driver_id, year), group in pdf.groupby(["driver_id", "year"]):

        # Ensure chronological order
        group = group.sort_values("round")

        # ------------------------------------------------------------------
        # IMPORTANT CHANGE:
        # ONLY USE EARLY/MID SEASON RACES
        # ------------------------------------------------------------------
        group = group.head(PREDICTION_ROUND)

        # Skip sequences with too little data
        if len(group) < 3:
            continue

        # Build feature sequence
        seq = group[available_feats].values.astype(np.float32)

        # ------------------------------------------------------------------
        # PAD / TRUNCATE
        # ------------------------------------------------------------------
        if len(seq) >= MAX_ROUNDS:
            seq = seq[:MAX_ROUNDS]

        else:
            pad = np.zeros(
                (MAX_ROUNDS - len(seq), feat_dim),
                dtype=np.float32
            )
            seq = np.vstack([seq, pad])

        # Championship label remains FULL-SEASON outcome
        label = int(group[CHAMP_TARGET_COL].max())

        sequences.append(seq)
        labels.append(label)

        meta.append({
            "driver_id":   driver_id,
            "driver_name": (
                group["driver_name"].iloc[0]
                if "driver_name" in group.columns
                else driver_id
            ),
            "year":  year,
            "label": label,
        })

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    meta_df = pd.DataFrame(meta)

    log.info(f"Built {len(X)} sequences")
    log.info(f"Champion sequences: {int(y.sum())}")
    log.info(f"Feature dimension: {feat_dim}")
    log.info(f"Sequence length: {MAX_ROUNDS}")

    return X, y, meta_df, available_feats


def normalize_sequences(X_train, X_test):
    """Normalize using train set statistics."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std
    return X_train_n, X_test_n


# ── Dataset ────────────────────────────────────────────────────────────────

class ChampDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Transformer Model ──────────────────────────────────────────────────────

class ChampionshipTransformer(nn.Module):
    """
    Input  : (batch, MAX_ROUNDS, feat_dim) — race-by-race stats per season
    Output : (batch, 1) — probability of winning the championship
    """
    def __init__(self, feat_dim, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        # Project input features to a dimension divisible by nhead
        base_dim = max(feat_dim * 2, nhead)
        self.d_model = ((base_dim + nhead - 1) // nhead) * nhead
        self.input_proj = nn.Linear(feat_dim, self.d_model)
        self.position_embedding = nn.Embedding(
            MAX_ROUNDS,
            self.d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        positions=torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x=x+ self.position_embedding(positions)             # (batch, rounds, d_model)
        x = self.transformer(x)            # (batch, rounds, d_model)
        x = x.mean(dim=1)                  # pool across rounds
        return self.classifier(x).squeeze(-1)


# ── Training ───────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([25.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_state   = None

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds    = model(X_batch)
                val_loss += criterion(preds, y_batch).item()

        avg_train = epoch_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{epochs} — Train: {avg_train:.4f}  Val: {avg_val:.4f}")

    model.load_state_dict(best_state)
    log.info(f"Best val loss: {best_val:.4f}")
    return model, train_losses, val_losses


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, meta_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    log.info(f"ROC-AUC: {auc:.4f}")
    log.info(f"Probability range:")
    log.info(f"Min:  {all_probs.min():.6f}")
    log.info(f"Max:  {all_probs.max():.6f}")
    log.info(f"Mean: {all_probs.mean():.6f}")

    top5_idx = np.argsort(all_probs)[-5:][::-1]

    log.info("\nTop 5 championship probabilities:")
    for idx in top5_idx:
        log.info(
            f"{meta_test.iloc[idx]['driver_name']} "
            f"({meta_test.iloc[idx]['year']}): "
            f"{all_probs[idx]:.4f}"
        )

    return all_probs, all_labels, auc


def predict_champions(all_probs, meta_test):
    """
    For each season in the test set, pick the driver with the
    highest championship probability as the predicted champion.
    """
    meta_test = meta_test.copy()
    meta_test["champ_prob"] = all_probs

    results = []
    for year, group in meta_test.groupby("year"):
        predicted = group.loc[group["champ_prob"].idxmax()]
        actual    = group[group["label"] == 1]

        predicted_name = predicted.get("driver_name", predicted.get("driver_id", "?"))
        actual_name    = actual["driver_name"].values[0] if len(actual) > 0 else "?"

        correct = int(predicted_name == actual_name)
        results.append({
            "year":      year,
            "predicted": predicted_name,
            "actual":    actual_name,
            "prob":      round(predicted["champ_prob"], 3),
            "correct":   correct,
        })

    results_df = pd.DataFrame(results).sort_values("year")
    accuracy   = results_df["correct"].mean()

    log.info(f"\nChampionship predictions:")
    log.info("\n" + results_df.to_string(index=False))
    log.info(f"\nChampionship prediction accuracy: {accuracy:.2%}")

    out_path = os.path.join(RESULTS_DIR, "championship_predictions.csv")
    results_df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    return results_df, accuracy

# ── Top K Accuracy ─────────────────────────────────────────────────────────

def top_k_accuracy(meta_test, probs, k=3):

    meta = meta_test.copy()
    meta["prob"] = probs

    correct = 0
    total = 0

    for year, group in meta.groupby("year"):

        topk = group.sort_values(
            "prob",
            ascending=False
        ).head(k)

        actual = group[group["label"] == 1]

        if len(actual) == 0:
            continue

        actual_driver = actual.iloc[0]["driver_name"]

        if actual_driver in topk["driver_name"].values:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

# ── Plots ──────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="steelblue")
    ax.plot(val_losses,   label="Val Loss",   color="orange")
    ax.set_title("Training Loss — Championship Transformer")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "champ_loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Loss curve saved to {out_path}")


def plot_championship_probs(results_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if c else "red" for c in results_df["correct"]]
    bars   = ax.bar(results_df["year"].astype(str), results_df["prob"], color=colors)
    ax.set_title("Championship Win Probability for Predicted Champion\n(green = correct, red = wrong)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Probability")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "champ_predictions_by_year.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Championship prediction plot saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    spark = get_spark()
    pdf   = load_champ_data(spark)
    spark.stop()

    # Build sequences
    X, y, meta_df, available_feats = build_sequences(pdf)
    feat_dim = len(available_feats)
    log.info(f"Using {feat_dim} features per round: {available_feats}")

    # Temporal split
    train_mask = meta_df["year"] <= TRAIN_YEARS_CUTOFF
    test_mask  = meta_df["year"] >  TRAIN_YEARS_CUTOFF

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    meta_train = meta_df[train_mask].reset_index(drop=True)
    meta_test  = meta_df[test_mask].reset_index(drop=True)

    # Validation split from train (last 5 years of train)
    val_mask   = meta_train["year"] >= (TRAIN_YEARS_CUTOFF - 4)
    X_val,  y_val  = X_train[val_mask],  y_train[val_mask]
    X_train2, y_train2 = X_train[~val_mask], y_train[~val_mask]

    log.info(f"Train: {len(X_train2)} sequences  Val: {len(X_val)}  Test: {len(X_test)}")

    # Normalize
    X_train2, X_val  = normalize_sequences(X_train2, X_val)
    _,         X_test = normalize_sequences(X_train2, X_test)

    # DataLoaders
    train_loader = DataLoader(ChampDataset(X_train2, y_train2),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ChampDataset(X_val,    y_val),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(ChampDataset(X_test,   y_test),
                              batch_size=BATCH_SIZE)

    # Model
    model = ChampionshipTransformer(feat_dim=feat_dim)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model, train_losses, val_losses = train_model(model, train_loader, val_loader)

    # Evaluate
    all_probs, all_labels, auc = evaluate_model(model, test_loader, meta_test)

    # Per-season champion predictions
    results_df, accuracy = predict_champions(all_probs, meta_test)
    top3 = top_k_accuracy(meta_test, all_probs, k=3)
    top5 = top_k_accuracy(meta_test, all_probs, k=5)

    log.info(f"Top-3 Championship Accuracy: {top3:.2%}")
    log.info(f"Top-5 Championship Accuracy: {top5:.2%}")

    # Plots
    plot_loss_curve(train_losses, val_losses)
    plot_championship_probs(results_df)

    # Save model
    model_path = os.path.join(RESULTS_DIR, "champ_transformer.pt")
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")

    log.info("=== Championship model complete ===")
    log.info(f"ROC-AUC:                {auc:.4f}")
    log.info(f"Championship accuracy:  {accuracy:.2%}")
    log.info(f"Probability range:")
    log.info(f"Min:  {all_probs.min():.6f}")
    log.info(f"Max:  {all_probs.max():.6f}")
    log.info(f"Mean: {all_probs.mean():.6f}")


if __name__ == "__main__":
    run()
