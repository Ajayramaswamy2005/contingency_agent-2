"""
EEQ401 — Deep Neural Network Grid Security Classification Engine on 100,000 Samples.

Trains a deep PyTorch classifier on all 1 lakh samples (dataset_100k.csv) using the 48GB GPU.
Evaluates Accuracy, Precision, Recall, Macro/Weighted F1-Scores, and Confusion Matrix.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

LABEL_MAP = {'Safe': 0, 'Alert': 1, 'Critical': 2}
INV_LABEL_MAP = {0: 'Safe', 1: 'Alert', 2: 'Critical'}

class GridSecurityClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def main():
    csv_path = "data/tabular/dataset_100k/dataset_100k.csv"
    save_dir = "models/classifier_100k"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "█" * 70)
    print("  EEQ401 — DEEP CLASSIFICATION TRAINING ON ALL 1 LAKH SAMPLES")
    print("█" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  [+] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Dataset
    t0 = time.time()
    print(f"  [+] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"  [+] Loaded {total_samples:,} samples in {time.time() - t0:.2f}s")

    # Fill non-convergent NaNs with physics-informed sentinels (matching state_builder.py)
    df['convergence_flag'] = df['convergence'].astype(float)
    df['min_bus_voltage'] = df['min_bus_voltage'].fillna(0.0)
    df['max_bus_voltage'] = df['max_bus_voltage'].fillna(0.0)
    df['max_line_loading'] = df['max_line_loading'].fillna(999.0)
    df['max_voltage_deviation'] = df['max_voltage_deviation'].fillna(1.0)
    median_total_load = float(df['total_load'].dropna().median() if not df['total_load'].dropna().empty else 259.0)
    df['total_load'] = df['total_load'].fillna(median_total_load)

    for c in df.columns:
        if c.startswith('V_bus'):
            df[c] = df[c].fillna(0.0)
        elif c.startswith('angle_bus'):
            df[c] = df[c].fillna(0.0)
        elif c.startswith('line_loading_'):
            df[c] = df[c].fillna(999.0)
        elif c.startswith('gen_P_'):
            df[c] = df[c].fillna(0.0)

    # Extract feature columns
    feature_cols = [c for c in df.columns if c.startswith(('V_bus', 'angle_bus', 'line_loading_', 'gen_P_')) 
                    or c in ['min_bus_voltage', 'max_bus_voltage', 'max_line_loading', 'max_voltage_deviation', 'total_load', 'convergence_flag']]
    
    # Also add one-hot for contingency element
    contingency_dummies = pd.get_dummies(df['contingency_element'], prefix='outage', dtype=float)
    
    X_cont = df[feature_cols].values
    X_outage = contingency_dummies.values
    X = np.hstack([X_cont, X_outage]).astype(np.float32)
    y = df['label'].map(LABEL_MAP).values.astype(np.int64)

    input_dim = X.shape[1]
    print(f"  [+] Input Feature Dimension: {input_dim}")
    print(f"  [+] Class Distribution: Safe: {(y==0).sum():,} | Alert: {(y==1).sum():,} | Critical: {(y==2).sum():,}")

    # 2. Train/Test Split (80% Train = 80k, 20% Test = 20k)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  [+] Split: Train = {len(X_train):,} samples | Test = {len(X_test):,} samples")

    # Scale continuous features
    scaler = StandardScaler()
    X_train[:, :len(feature_cols)] = scaler.fit_transform(X_train[:, :len(feature_cols)])
    X_test[:, :len(feature_cols)] = scaler.transform(X_test[:, :len(feature_cols)])

    # PyTorch DataLoaders
    batch_size = 512
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 3. Model, Loss, Optimizer
    model = GridSecurityClassifier(input_dim=input_dim, hidden_dim=256, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 15

    print(f"\n[+] Starting GPU Training for {epochs} Epochs (Batch Size = {batch_size})...")
    t_train_start = time.time()
    history = []

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(by)
                preds = out.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"  Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        history.append({
            'epoch': ep + 1,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc * 100, 2),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc * 100, 2),
        })

    train_duration = time.time() - t_train_start
    print(f"  [+] Training finished in {train_duration:.2f} seconds!")

    # 4. Final Comprehensive Evaluation on Test Set (20,000 cases)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            out = model(bx)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    macro_prec = precision_score(all_targets, all_preds, average='macro')
    macro_rec = recall_score(all_targets, all_preds, average='macro')

    # Per-class metrics
    f1_per_class = f1_score(all_targets, all_preds, average=None)
    prec_per_class = precision_score(all_targets, all_preds, average=None)
    rec_per_class = recall_score(all_targets, all_preds, average=None)

    cm = confusion_matrix(all_targets, all_preds)

    metrics = {
        'total_samples': total_samples,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'training_time_seconds': round(train_duration, 2),
        'accuracy': round(float(acc * 100), 2),
        'macro_f1': round(float(macro_f1 * 100), 2),
        'weighted_f1': round(float(weighted_f1 * 100), 2),
        'macro_precision': round(float(macro_prec * 100), 2),
        'macro_recall': round(float(macro_rec * 100), 2),
        'per_class': {
            'Safe': {
                'precision': round(float(prec_per_class[0] * 100), 2),
                'recall': round(float(rec_per_class[0] * 100), 2),
                'f1_score': round(float(f1_per_class[0] * 100), 2),
                'support': int((all_targets == 0).sum())
            },
            'Alert': {
                'precision': round(float(prec_per_class[1] * 100), 2),
                'recall': round(float(rec_per_class[1] * 100), 2),
                'f1_score': round(float(f1_per_class[1] * 100), 2),
                'support': int((all_targets == 1).sum())
            },
            'Critical': {
                'precision': round(float(prec_per_class[2] * 100), 2),
                'recall': round(float(rec_per_class[2] * 100), 2),
                'f1_score': round(float(f1_per_class[2] * 100), 2),
                'support': int((all_targets == 2).sum())
            }
        },
        'confusion_matrix': cm.tolist(),
        'history': history
    }

    # Save Model Weights and Metrics
    model_save_path = os.path.join(save_dir, "grid_security_classifier.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"  [+] Saved Classifier Weights to: {model_save_path}")

    metrics_save_path = os.path.join(save_dir, "classification_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [+] Saved Classification Metrics to: {metrics_save_path}")

    print("\n" + "═" * 70)
    print("  FINAL 1 LAKH SAMPLES CLASSIFIER EVALUATION REPORT")
    print("═" * 70)
    print(f"  Overall Accuracy:    {metrics['accuracy']}%")
    print(f"  Macro F1-Score:      {metrics['macro_f1']}%")
    print(f"  Weighted F1-Score:   {metrics['weighted_f1']}%")
    print(f"  Macro Precision:     {metrics['macro_precision']}%")
    print(f"  Macro Recall:        {metrics['macro_recall']}%")
    print("  -------------------------------------------------------------")
    print(f"  Safe Class:          Precision {metrics['per_class']['Safe']['precision']}% | Recall {metrics['per_class']['Safe']['recall']}% | F1 {metrics['per_class']['Safe']['f1_score']}%")
    print(f"  Alert Class:         Precision {metrics['per_class']['Alert']['precision']}% | Recall {metrics['per_class']['Alert']['recall']}% | F1 {metrics['per_class']['Alert']['f1_score']}%")
    print(f"  Critical Class:      Precision {metrics['per_class']['Critical']['precision']}% | Recall {metrics['per_class']['Critical']['recall']}% | F1 {metrics['per_class']['Critical']['f1_score']}%")
    print("  -------------------------------------------------------------")
    print(f"  Confusion Matrix (Rows=True, Cols=Pred):")
    print(f"  Safe:     {cm[0]}")
    print(f"  Alert:    {cm[1]}")
    print(f"  Critical: {cm[2]}")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    main()
