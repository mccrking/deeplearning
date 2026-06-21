"""
Modèle MLP (Perceptron Multicouche)
Projet Deep Learning - Mehdi Chmiti - 4IAD G3
Dataset : Breast Cancer Wisconsin (classification binaire)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.metrics_logger import (
    train_model, evaluate, plot_learning_curves, 
    plot_confusion_matrix, plot_roc_curves, generate_classification_report
)


class MLP(nn.Module):
    """
    Perceptron Multicouche avec architecture en entonnoir.
    Structure : Input → 128 → 64 → 32 → Output
    Avec BatchNormalization et Dropout pour la régularisation.
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def run_mlp_experiment():
    """Exécute l'expérimentation complète du MLP."""
    print("\n" + "="*70)
    print("  EXPERIMENTATION MLP - Breast Cancer Wisconsin")
    print("="*70)
    
    # === 1. Chargement et exploration des données ===
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names  # ['malignant', 'benign']
    
    print(f"\nDataset Breast Cancer Wisconsin :")
    print(f"  Nombre d'échantillons : {X.shape[0]}")
    print(f"  Nombre de features : {X.shape[1]}")
    print(f"  Classes : {list(class_names)}")
    print(f"  Distribution des classes : Malignant={sum(y==0)}, Benign={sum(y==1)}")
    print(f"  Ratio de déséquilibre : {sum(y==1)/sum(y==0):.2f}")
    
    # Statistiques descriptives
    print(f"\n  Statistiques descriptives :")
    print(f"  Moyenne des features (min-max) : {X.mean(axis=0).min():.2f} - {X.mean(axis=0).max():.2f}")
    print(f"  Écart-type des features (min-max) : {X.std(axis=0).min():.2f} - {X.std(axis=0).max():.2f}")
    
    # === 2. Prétraitement ===
    # Split stratifié : 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/(1-0.15), random_state=SEED, stratify=y_temp
    )
    
    print(f"\n  Split des données :")
    print(f"  Train : {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  Val   : {X_val.shape[0]} ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  Test  : {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    
    # Standardisation (fit sur train uniquement !)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Conversion en tenseurs PyTorch
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=MLP_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MLP_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=MLP_CONFIG['batch_size'], shuffle=False)
    
    # === 3. Définition du modèle ===
    set_seed()
    model = MLP(
        input_size=X.shape[1],
        hidden_sizes=MLP_CONFIG['hidden_sizes'],
        num_classes=2,
        dropout=MLP_CONFIG['dropout']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Modèle MLP :")
    print(f"  Paramètres totaux : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}")
    print(f"  Architecture : {X.shape[1]} → {' → '.join(map(str, MLP_CONFIG['hidden_sizes']))} → 2")
    
    # === 4. Entraînement ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=MLP_CONFIG['learning_rate'],
        weight_decay=MLP_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
    )
    
    logger, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        DEVICE, MLP_CONFIG['epochs'], MLP_CONFIG['patience'],
        model_name="MLP"
    )
    
    # === 5. Évaluation finale sur le test set ===
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n{'='*50}")
    print(f"  RÉSULTATS FINAUX MLP (Test Set)")
    print(f"{'='*50}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    
    # === 6. Visualisations ===
    plot_learning_curves(logger, "MLP", PLOTS_DIR)
    plot_confusion_matrix(test_labels, test_preds, ['Malin', 'Benin'], "MLP", PLOTS_DIR)
    aucs = plot_roc_curves(test_labels, test_probs, 2, ['Malin', 'Benin'], "MLP", PLOTS_DIR)
    report = generate_classification_report(test_labels, test_preds, ['Malin', 'Benin'], "MLP")
    
    # === 7. Permutation Feature Importance ===
    try:
        compute_permutation_importance(model, X_test, y_test, feature_names, scaler)
    except Exception as e:
        print(f"  Permutation importance skip : {e}")
    
    # Sauvegarde des résultats
    results = {
        'model': 'MLP',
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_f1),
        'best_epoch': int(best_epoch),
        'total_params': int(total_params),
        'auc_per_class': [float(a) for a in aucs] if aucs else [],
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'MLP_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def compute_permutation_importance(model, X_test, y_test, feature_names, scaler):
    """Calcule l'importance des features par permutation."""
    from sklearn.metrics import f1_score
    
    model.eval()
    
    # Score de base
    X_test_tensor = torch.FloatTensor(scaler.transform(X_test) if False else torch.FloatTensor(X_test)).to(DEVICE)
    y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        base_preds = torch.argmax(outputs, dim=1).cpu().numpy()
    base_f1 = f1_score(y_test, base_preds, average='macro')
    
    # Permutation
    importances = []
    X_test_np = X_test.copy()
    
    for i in range(X_test_np.shape[1]):
        X_permuted = X_test_np.copy()
        np.random.shuffle(X_permuted[:, i])
        X_permuted_scaled = scaler.transform(X_permuted)
        X_perm_tensor = torch.FloatTensor(X_permuted_scaled).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(X_perm_tensor)
            perm_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        perm_f1 = f1_score(y_test, perm_preds, average='macro')
        importances.append(base_f1 - perm_f1)
    
    # Afficher top 10 features
    indices = np.argsort(importances)[::-1]
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    top_k = min(10, len(feature_names))
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), [importances[i] for i in indices[:top_k]], color='steelblue')
    plt.yticks(range(top_k), [feature_names[i] for i in indices[:top_k]])
    plt.xlabel("Importance (baisse de F1 par permutation)")
    plt.title("Permutation Feature Importance - MLP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'MLP_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance sauvegardée : MLP_feature_importance.png")


if __name__ == "__main__":
    results = run_mlp_experiment()
