"""
Architecture Hybride : CNN + LSTM
Projet Deep Learning - Mehdi Chmiti - 4IAD G3
Dataset : UCI HAR (Human Activity Recognition)
  - Signaux multicanaux (9 canaux) = structure spatiale
  - Séquences temporelles (128 pas de temps) = dynamique temporelle
  - Le CNN extrait les features par canal, le LSTM capture les dépendances temporelles
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.metrics_logger import (
    train_model, evaluate, plot_learning_curves,
    plot_confusion_matrix, plot_roc_curves, generate_classification_report
)


class CNNLSTM(nn.Module):
    """
    Architecture hybride CNN + LSTM.
    Le CNN extrait les features spatiales par timestep,
    puis le LSTM capture les dépendances temporelles.
    
    Pipeline : Entrée (batch, seq_len, channels) 
               → CNN 1D par timestep → (batch, seq_len, cnn_features)
               → LSTM → (batch, hidden) 
               → Dense → Prédiction
    """
    def __init__(self, input_channels, cnn_filters, lstm_hidden, lstm_layers,
                 num_classes, dropout=0.3, bidirectional=True):
        super(CNNLSTM, self).__init__()
        
        # Extracteur CNN 1D (features spatiales)
        cnn_layers = []
        prev_ch = input_channels
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(prev_ch, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            prev_ch = filters
        
        self.cnn_extractor = nn.Sequential(*cnn_layers)
        
        # Calcul de la taille de sortie du CNN
        # Après N blocs avec MaxPool1d(2), la dimension temporelle est divisée par 2^N
        # On utilise AdaptiveAvgPool1d pour fixer la sortie
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        cnn_out_features = cnn_filters[-1]
        
        # LSTM (dépendances temporelles)
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifieur
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x : (batch, seq_len, channels)
        batch_size, seq_len, channels = x.size()
        
        # Appliquer le CNN 1D sur chaque timestep
        # Reshape : (batch * seq_len, channels, 1) pour Conv1d
        x_cnn = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        # CNN 1D sur toute la séquence
        cnn_features = self.cnn_extractor(x_cnn)  # (batch, filters, reduced_seq)
        cnn_features = self.adaptive_pool(cnn_features)  # (batch, filters, 1)
        cnn_features = cnn_features.squeeze(-1)  # (batch, filters)
        
        # Pour avoir une dimension temporelle, on reshape
        # On traite le signal complet comme une seule séquence
        cnn_features = cnn_features.unsqueeze(1)  # (batch, 1, filters)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)  # (batch, 1, hidden*2)
        
        # Prendre la dernière sortie
        if self.lstm.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], 
                                  lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        
        return self.classifier(lstm_out)


class CNNLSTMSequential(nn.Module):
    """
    Architecture hybride CNN + LSTM séquentielle.
    Découpe le signal en fenêtres, extrait les features CNN par fenêtre,
    puis les alimente dans le LSTM.
    """
    def __init__(self, input_channels, seq_len, window_size, cnn_filters,
                 lstm_hidden, lstm_layers, num_classes, dropout=0.3, bidirectional=True):
        super(CNNLSTMSequential, self).__init__()
        
        self.window_size = window_size
        self.n_windows = seq_len // window_size
        
        # CNN par fenêtre
        cnn_layers = []
        prev_ch = input_channels
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(prev_ch, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            prev_ch = filters
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        cnn_out_features = cnn_filters[-1]
        
        # LSTM sur les features des fenêtres
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x : (batch, seq_len, channels)
        batch_size = x.size(0)
        channels = x.size(2)
        
        # Découper en fenêtres
        # x : (batch, n_windows, window_size, channels)
        x_windows = x[:, :self.n_windows * self.window_size, :]
        x_windows = x_windows.view(batch_size, self.n_windows, self.window_size, channels)
        
        # CNN sur chaque fenêtre
        cnn_features_list = []
        for w in range(self.n_windows):
            window = x_windows[:, w, :, :].permute(0, 2, 1)  # (batch, channels, window_size)
            cnn_out = self.cnn(window)
            cnn_out = self.adaptive_pool(cnn_out).squeeze(-1)  # (batch, filters)
            cnn_features_list.append(cnn_out)
        
        # Stack : (batch, n_windows, filters)
        cnn_features = torch.stack(cnn_features_list, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        if self.lstm.bidirectional:
            out = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size],
                            lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        else:
            out = lstm_out[:, -1, :]
        
        return self.classifier(out)


class CNNMLPFusion(nn.Module):
    """
    Architecture hybride CNN + MLP (fusion multimodale).
    Combine des features spatiales (CNN) et tabulaires (MLP).
    Late fusion : concaténation des embeddings avant la décision.
    """
    def __init__(self, tab_input_size, cnn_filters, mlp_hidden, num_classes, dropout=0.3):
        super(CNNMLPFusion, self).__init__()
        
        # Branche CNN (signaux)
        cnn_layers = []
        prev_ch = 1
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(prev_ch, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            prev_ch = filters
        
        self.cnn_branch = nn.Sequential(*cnn_layers)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        cnn_embed_size = cnn_filters[-1]
        
        # Branche MLP (données tabulaires)
        self.mlp_branch = nn.Sequential(
            nn.Linear(tab_input_size, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU()
        )
        mlp_embed_size = mlp_hidden // 2
        
        # Fusion (late fusion)
        fusion_size = cnn_embed_size + mlp_embed_size
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x_signal, x_tabular):
        # Branche CNN
        cnn_out = self.cnn_branch(x_signal)
        cnn_out = self.cnn_pool(cnn_out).squeeze(-1)  # (batch, cnn_embed_size)
        
        # Branche MLP
        mlp_out = self.mlp_branch(x_tabular)  # (batch, mlp_embed_size)
        
        # Fusion
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fusion_classifier(fused)


def generate_hybrid_data(n_samples=4000, seq_len=128, n_channels=3, n_classes=4):
    """Génère des données spatio-temporelles pour les architectures hybrides."""
    X = np.zeros((n_samples, seq_len, n_channels))
    y = np.zeros(n_samples, dtype=int)
    
    samples_per_class = n_samples // n_classes
    t = np.linspace(0, 6 * np.pi, seq_len)
    
    idx = 0
    patterns = [
        lambda t: np.sin(t),                # Sinus
        lambda t: np.cos(2 * t),            # Cosinus double fréquence
        lambda t: np.sign(np.sin(t)),       # Signal carré
        lambda t: 2 * (t / (2*np.pi) % 1) - 1  # Dents de scie
    ]
    
    for class_idx, pattern in enumerate(patterns):
        for i in range(samples_per_class):
            base = pattern(t * np.random.uniform(0.8, 1.2))
            amplitude = np.random.uniform(0.5, 1.5)
            for ch in range(n_channels):
                freq_mod = np.random.uniform(0.9, 1.1)
                noise = np.random.normal(0, 0.15, seq_len)
                X[idx, :, ch] = amplitude * pattern(t * freq_mod + ch * 0.5) + noise
            y[idx] = class_idx
            idx += 1
    
    return X, y


def run_hybrid_experiment():
    """Exécute l'expérimentation avec l'architecture hybride CNN+LSTM sur UCI HAR."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENTATION HYBRIDE CNN+LSTM - UCI HAR")
    print(f"{'='*70}")
    
    # Réutilisation du chargeur UCI HAR défini dans rnn_models.py
    from models.rnn_models import load_uci_har_inertial, prepare_har_data, UCI_HAR_CLASS_NAMES
    
    CLASS_NAMES = UCI_HAR_CLASS_NAMES  # 6 activités
    NUM_CLASSES = 6
    INPUT_CHANNELS = 9   # 9 canaux Inertial Signals
    
    # === 1. Données UCI HAR ===
    X_train, y_train, X_test, y_test = load_uci_har_inertial()
    print(f"\n  Dataset UCI HAR :")
    print(f"  Train : {X_train.shape[0]}, Test : {X_test.shape[0]}")
    print(f"  Séquence : {X_train.shape[1]} pas de temps, Canaux : {X_train.shape[2]}")
    print(f"  Distribution (train) : {[int(sum(y_train==i)) for i in range(NUM_CLASSES)]}")
    
    train_loader, val_loader, test_loader = prepare_har_data(
        X_train, y_train, X_test, y_test, batch_size=HYBRID_CONFIG['batch_size']
    )
    
    # === 2. Modèle CNN+LSTM ===
    set_seed()
    model = CNNLSTM(
        input_channels=INPUT_CHANNELS,
        cnn_filters=HYBRID_CONFIG['cnn_filters'],
        lstm_hidden=HYBRID_CONFIG['lstm_hidden'],
        lstm_layers=HYBRID_CONFIG['lstm_layers'],
        num_classes=NUM_CLASSES,
        dropout=HYBRID_CONFIG['dropout'],
        bidirectional=True
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Modèle CNN+LSTM :")
    print(f"  Paramètres : {total_params:,}")
    
    # === 3. Entraînement ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HYBRID_CONFIG['learning_rate'],
        weight_decay=HYBRID_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, 
    )
    
    logger, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        DEVICE, HYBRID_CONFIG['epochs'], HYBRID_CONFIG['patience'],
        gradient_clip=HYBRID_CONFIG['gradient_clip'],
        model_name="CNN_LSTM"
    )
    
    # === 4. Évaluation ===
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n{'='*50}")
    print(f"  RÉSULTATS FINAUX CNN+LSTM (Test Set)")
    print(f"{'='*50}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    
    # === 5. Visualisations ===
    plot_learning_curves(logger, "CNN_LSTM", PLOTS_DIR)
    plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES, "CNN_LSTM", PLOTS_DIR)
    try:
        aucs = plot_roc_curves(test_labels, test_probs, 4, CLASS_NAMES, "CNN_LSTM", PLOTS_DIR)
    except:
        aucs = []
    report = generate_classification_report(test_labels, test_preds, CLASS_NAMES, "CNN_LSTM")
    
    # Sauvegarde
    import json
    results = {
        'model': 'CNN_LSTM',
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_f1),
        'best_epoch': int(best_epoch),
        'total_params': int(total_params),
    }
    
    with open(os.path.join(RESULTS_DIR, 'CNN_LSTM_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_hybrid_experiment()
