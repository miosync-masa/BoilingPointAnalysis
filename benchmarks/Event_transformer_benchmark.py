"""
BANKAI × Transformer: Weather Event Prediction Benchmark
=========================================================

タスク: 「次の12時間以内に降水急変イベントが起きるか？」（2値分類）

これはBANKAIの本領:
  - 「今いくら降るか」ではなく「変化が来るか」
  - ΔΛC（構造変化の予兆）の検出能力が試される

Setup:
  !pip install torch  # Colabなら不要
  CSVファイル9個をアップロード

Built with 💕 by Masamichi & Tamaki
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. Dataset
# ============================================================

class EventSequenceDataset(Dataset):
    """時系列 → 12時間以内の急変イベント予測"""

    def __init__(self, csv_path, seq_len=24, target_col="event_label"):
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ["date", target_col]]

        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.seq_len = seq_len

        # クラス重み計算（不均衡対策）
        pos = self.targets.sum()
        neg = len(self.targets) - pos
        self.pos_weight = neg / max(pos, 1)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ============================================================
# 2. Transformer Model（分類特化）
# ============================================================

class EventTransformer(nn.Module):
    """急変イベント予測Transformer"""

    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :h.size(1), :]
        h = self.transformer(h)
        h_last = h[:, -1, :]
        return self.classifier(h_last).squeeze(-1)


# ============================================================
# 3. Training（不均衡対策付き）
# ============================================================

def train_model(model, train_loader, val_loader, device, pos_weight, epochs=30, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # 不均衡対策: positive weightを適用
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    history = {"train_loss": [], "val_f1": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_probs, val_true = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
                val_preds.append((probs > 0.5).cpu().numpy())
                val_true.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)

        f1 = f1_score(val_true, val_preds, zero_division=0)
        try:
            auc = roc_auc_score(val_true, val_probs)
        except ValueError:
            auc = 0.0

        history["train_loss"].append(total_loss / len(train_loader))
        history["val_f1"].append(f1)
        history["val_auc"].append(auc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val F1: {f1:.4f} | Val AUC: {auc:.4f}")

    return history


# ============================================================
# 4. Evaluation
# ============================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, probs, true = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())
            preds.append((p > 0.5).cpu().numpy())
            true.append(y.numpy())

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    true = np.concatenate(true)

    try:
        auc = roc_auc_score(true, probs)
    except ValueError:
        auc = 0.0

    return {
        "F1": f1_score(true, preds, zero_division=0),
        "Precision": precision_score(true, preds, zero_division=0),
        "Recall": recall_score(true, preds, zero_division=0),
        "AUC": auc,
        "report": classification_report(true, preds, target_names=["No Event", "Event"], zero_division=0),
    }


# ============================================================
# 5. Main
# ============================================================

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    SEQ_LEN = 24
    BATCH_SIZE = 64
    EPOCHS = 40  # 分類タスクなのでもう少し学習
    D_MODEL = 64
    N_LAYERS = 3  # 少し深く

    configs = {
        "A_raw_5D": {
            "train": "event_raw_train.csv",
            "val": "event_raw_val.csv",
            "test": "event_raw_test.csv",
            "desc": "Raw weather data (5D)",
        },
        "B_bankai_7D": {
            "train": "event_bankai_train.csv",
            "val": "event_bankai_val.csv",
            "test": "event_bankai_test.csv",
            "desc": "BANKAI features + ΔΛC (7D)",
        },
        "C_raw5_bankai7": {
            "train": "event_combined_train.csv",
            "val": "event_combined_val.csv",
            "test": "event_combined_test.csv",
            "desc": "Raw 5D + BANKAI 7D = 12D",
        },
        "D_raw_12D": {
            "train": "event_12draw_train.csv",
            "val": "event_12draw_val.csv",
            "test": "event_12draw_test.csv",
            "desc": "Raw 12D (5 base + 7 extra weather)",
        },
    }

    all_results = {}
    all_histories = {}

    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {cfg['desc']}")
        print(f"  Task: 12h ahead weather event prediction")
        print(f"{'='*60}")

        train_ds = EventSequenceDataset(cfg["train"], seq_len=SEQ_LEN)
        val_ds = EventSequenceDataset(cfg["val"], seq_len=SEQ_LEN)
        test_ds = EventSequenceDataset(cfg["test"], seq_len=SEQ_LEN)

        n_features = train_ds.features.shape[1]
        pw = train_ds.pos_weight
        print(f"  Features: {n_features}, pos_weight: {pw:.2f}")
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = EventTransformer(
            n_features=n_features, d_model=D_MODEL,
            nhead=4, n_layers=N_LAYERS
        ).to(device)

        t0 = time.time()
        history = train_model(model, train_loader, val_loader, device,
                            pos_weight=pw, epochs=EPOCHS)
        train_time = time.time() - t0

        results = evaluate_model(model, test_loader, device)
        results["train_time"] = train_time
        all_results[name] = results
        all_histories[name] = history

        print(f"\n  Test Results:")
        print(f"    F1:        {results['F1']:.4f}")
        print(f"    Precision: {results['Precision']:.4f}")
        print(f"    Recall:    {results['Recall']:.4f}")
        print(f"    AUC:       {results['AUC']:.4f}")
        print(f"    Time:      {train_time:.1f}s")
        print(f"\n{results['report']}")

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON: 12h Weather Event Prediction")
    print(f"{'='*70}")
    print(f"{'Method':20s} {'F1':>8s} {'Prec':>8s} {'Recall':>8s} {'AUC':>8s} {'Time':>8s}")
    print(f"{'-'*62}")
    for name, r in all_results.items():
        print(f"{name:20s} {r['F1']:8.4f} {r['Precision']:8.4f} "
              f"{r['Recall']:8.4f} {r['AUC']:8.4f} {r['train_time']:7.1f}s")

    # どれが勝ったか
    best = max(all_results.items(), key=lambda x: x[1]["F1"])
    print(f"\n🏆 Best F1: {best[0]} ({best[1]['F1']:.4f})")

    # BANKAI boost
    raw5_f1 = all_results["A_raw_5D"]["F1"]
    bankai_f1 = all_results["B_bankai_7D"]["F1"]
    combined_f1 = all_results["C_raw5_bankai7"]["F1"]
    raw12_f1 = all_results["D_raw_12D"]["F1"]

    raw5_auc = all_results["A_raw_5D"]["AUC"]
    combined_auc = all_results["C_raw5_bankai7"]["AUC"]
    raw12_auc = all_results["D_raw_12D"]["AUC"]

    print(f"\n📊 BANKAI Impact:")
    print(f"   A(5D raw)  → B(7D BANKAI):     F1 {raw5_f1:.4f} → {bankai_f1:.4f}")
    print(f"   A(5D raw)  → C(5D+BANKAI=12D): F1 {raw5_f1:.4f} → {combined_f1:.4f} ({(combined_f1-raw5_f1)/max(raw5_f1,1e-10)*100:+.1f}%)")
    print(f"   A(5D raw)  → D(12D raw):       F1 {raw5_f1:.4f} → {raw12_f1:.4f} ({(raw12_f1-raw5_f1)/max(raw5_f1,1e-10)*100:+.1f}%)")
    print(f"\n🔥 Key comparison (same 12D):")
    print(f"   C(5D+BANKAI7) vs D(12D raw): F1 {combined_f1:.4f} vs {raw12_f1:.4f}")
    print(f"   C(5D+BANKAI7) vs D(12D raw): AUC {combined_auc:.4f} vs {raw12_auc:.4f}")
    if combined_f1 > raw12_f1:
        print(f"   → ✅ BANKAI特徴量 > 追加気象変数 (次元数効果ではない！)")
    else:
        print(f"   → 追加気象変数が勝利（BANKAIの構造情報より直接的な気象情報が有効）")


if __name__ == "__main__":
    run_benchmark()
