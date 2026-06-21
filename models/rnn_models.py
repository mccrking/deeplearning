"""
Modèles RNN, LSTM et GRU
Projet Deep Learning - Mehdi Chmiti - 4IAD G3
Dataset : UCI HAR (Human Activity Recognition with Smartphones)
  - 30 volontaires, smartphone à la ceinture
  - Accéléromètre + Gyroscope 3 axes (9 canaux)
  - 6 activités : WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING
  - 10 299 séquences de 128 pas de temps
Source : UCI ML Repository (aussi disponible sur Kaggle)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.metrics_logger import (
    train_model, evaluate, plot_learning_curves,
    plot_confusion_matrix, plot_roc_curves, generate_classification_report
)


# ============================================================
# Modèles
# ============================================================

class SimpleRNN(nn.Module):
    """RNN simple pour classification de séquences."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 bidirectional=True, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes)
        )
    
    def forward(self, x):
        # x shape : (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Prendre le dernier timestep
        if self.bidirectional:
            # Concaténer forward et backward du dernier timestep
            out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            out = out[:, -1, :]
        return self.classifier(out)


class LSTMModel(nn.Module):
    """LSTM pour classification de séquences longues."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 bidirectional=True, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes)
        )
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        if self.bidirectional:
            out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            out = out[:, -1, :]
        return self.classifier(out)


class GRUModel(nn.Module):
    """GRU - Alternative légère au LSTM."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 bidirectional=True, dropout=0.3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, num_classes)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        if self.bidirectional:
            out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            out = out[:, -1, :]
        return self.classifier(out)


# ============================================================
# Chargement du dataset UCI HAR (réel)
# ============================================================

UCI_HAR_DIR = "/home/z/my-project/download/deep_learning_project/data/uci_har/UCI HAR Dataset"
UCI_HAR_CLASS_NAMES = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 'SITTING', 'STANDING', 'LAYING']


def load_uci_har_inertial():
    """
    Charge les signaux bruts Inertial Signals du dataset UCI HAR.
    
    9 canaux par échantillon :
      - body_acc_x/y/z   (accélération du corps, 3 axes)
      - body_gyro_x/y/z  (vitesse angulaire, 3 axes)
      - total_acc_x/y/z  (accélération totale, 3 axes)
    
    Séquence de 128 pas de temps (échantillonnage à 50 Hz, fenêtre de 2.56 s).
    6 classes : WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING.
    
    Retourne :
      X_train : (7352, 128, 9)
      y_train : (7352,)
      X_test  : (2947, 128, 9)
      y_test  : (2947,)
    """
    channels = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    def _load_split(split):
        folder = os.path.join(UCI_HAR_DIR, split, 'Inertial Signals')
        signals = []
        for ch in channels:
            fname = f'{ch}_{split}.txt'
            arr = np.loadtxt(os.path.join(folder, fname))
            signals.append(arr)
        # Stack en (N, 128, 9)
        X = np.stack(signals, axis=-1)
        y = np.loadtxt(os.path.join(UCI_HAR_DIR, split, f'y_{split}.txt')).astype(int) - 1
        return X, y
    
    X_train, y_train = _load_split('train')
    X_test, y_test = _load_split('test')
    return X_train, y_train, X_test, y_test


def prepare_har_data(X_train, y_train, X_test, y_test, batch_size=64, val_ratio=0.15):
    """
    Prépare les DataLoaders PyTorch pour UCI HAR.
    - Extrait un set de validation stratifié depuis le train
    - Normalise par canal (z-score calculé sur train uniquement)
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    
    # Normalisation par canal
    n_features = X_tr.shape[2]
    for f in range(n_features):
        mean = X_tr[:, :, f].mean()
        std = X_tr[:, :, f].std()
        X_tr[:, :, f] = (X_tr[:, :, f] - mean) / (std + 1e-8)
        X_val[:, :, f] = (X_val[:, :, f] - mean) / (std + 1e-8)
        X_test[:, :, f] = (X_test[:, :, f] - mean) / (std + 1e-8)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def prepare_rnn_data(X, y, batch_size=64):
    """Prépare les DataLoaders pour les modèles récurrents."""
    # Split stratifié
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/(1-0.15), random_state=SEED, stratify=y_temp
    )
    
    # Normalisation par canal (z-score)
    n_features = X.shape[2]
    for f in range(n_features):
        mean = X_train[:, :, f].mean()
        std = X_train[:, :, f].std()
        X_train[:, :, f] = (X_train[:, :, f] - mean) / (std + 1e-8)
        X_val[:, :, f] = (X_val[:, :, f] - mean) / (std + 1e-8)
        X_test[:, :, f] = (X_test[:, :, f] - mean) / (std + 1e-8)
    
    # Tenseurs PyTorch
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================
# Expérimentations
# ============================================================

def run_rnn_experiment(model_class, model_name):
    """Exécute une expérimentation avec un modèle récurrent sur UCI HAR."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENTATION {model_name} - UCI HAR (Human Activity Recognition)")
    print(f"{'='*70}")
    
    CLASS_NAMES = UCI_HAR_CLASS_NAMES  # 6 activités
    NUM_CLASSES = 6
    INPUT_SIZE = 9   # 9 canaux Inertial Signals
    
    # === 1. Chargement du dataset réel UCI HAR ===
    X_train, y_train, X_test, y_test = load_uci_har_inertial()
    print(f"\n  Dataset UCI HAR :")
    print(f"  Train : {X_train.shape[0]}, Test : {X_test.shape[0]}")
    print(f"  Séquence : {X_train.shape[1]} pas de temps, Features : {X_train.shape[2]} canaux")
    print(f"  Distribution (train) : {[int(sum(y_train==i)) for i in range(NUM_CLASSES)]}")
    
    train_loader, val_loader, test_loader = prepare_har_data(
        X_train, y_train, X_test, y_test, batch_size=RNN_CONFIG['batch_size']
    )
    
    # === 2. Modèle ===
    set_seed()
    model = model_class(
        input_size=INPUT_SIZE,
        hidden_size=RNN_CONFIG['hidden_size'],
        num_layers=RNN_CONFIG['num_layers'],
        num_classes=NUM_CLASSES,
        bidirectional=RNN_CONFIG['bidirectional'],
        dropout=RNN_CONFIG['dropout']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Modèle {model_name} :")
    print(f"  Paramètres : {total_params:,}")
    print(f"  Hidden size : {RNN_CONFIG['hidden_size']}")
    print(f"  Couches : {RNN_CONFIG['num_layers']}")
    print(f"  Bidirectionnel : {RNN_CONFIG['bidirectional']}")
    
    # === 3. Entraînement ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=RNN_CONFIG['learning_rate'],
        weight_decay=RNN_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
    )
    
    logger, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        DEVICE, RNN_CONFIG['epochs'], RNN_CONFIG['patience'],
        gradient_clip=RNN_CONFIG['gradient_clip'],
        model_name=model_name
    )
    
    # === 4. Évaluation finale ===
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n{'='*50}")
    print(f"  RÉSULTATS FINAUX {model_name} (Test Set)")
    print(f"{'='*50}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    
    # === 5. Visualisations ===
    plot_learning_curves(logger, model_name, PLOTS_DIR)
    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES, model_name, PLOTS_DIR)
    try:
        aucs = plot_roc_curves(test_labels, test_probs, NUM_CLASSES, CLASS_NAMES, model_name, PLOTS_DIR)
    except:
        aucs = []
    report = generate_classification_report(test_labels, test_preds, CLASS_NAMES, model_name)
    
    # Sauvegarde
    import json
    results = {
        'model': model_name,
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_f1),
        'best_epoch': int(best_epoch),
        'total_params': int(total_params),
    }
    
    with open(os.path.join(RESULTS_DIR, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_all_rnn_experiments():
    """Exécute les expérimentations pour RNN, LSTM et GRU."""
    results = {}
    
    # RNN Simple
    results['RNN'] = run_rnn_experiment(SimpleRNN, "RNN")
    
    # LSTM
    results['LSTM'] = run_rnn_experiment(LSTMModel, "LSTM")
    
    # GRU
    results['GRU'] = run_rnn_experiment(GRUModel, "GRU")
    
    # === Comparaison RNN / LSTM / GRU ===
    plot_rnn_comparison(results)
    
    return results


def plot_rnn_comparison(results):
    """Trace un graphique comparatif des modèles récurrents sur UCI HAR."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    
    models = list(results.keys())
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[model][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metriques')
    ax.set_ylabel('Score')
    ax.set_title('Comparaison RNN / LSTM / GRU sur UCI HAR (6 activites)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(PLOTS_DIR, 'RNN_comparison.png'), dpi=150)
    plt.close()
    print(f"\n  Comparaison RNN/LSTM/GRU sauvegardée : RNN_comparison.png")


if __name__ == "__main__":
    results = run_all_rnn_experiments()
