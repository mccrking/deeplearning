"""
Utilitaires d'entraînement, métriques et visualisation
Projet Deep Learning - Mehdi Chmiti - 4IAD G3
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import seaborn as sns
import os
import json

# Configuration matplotlib pour le français
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
except:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Sarasa Mono SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class MetricsLogger:
    """Enregistre les métriques d'entraînement par époque."""
    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'lr': []
        }
    
    def log(self, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, lr):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_f1'].append(train_f1)
        self.history['val_f1'].append(val_f1)
        self.history['lr'].append(lr)
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_epoch(model, dataloader, criterion, optimizer, device, gradient_clip=None):
    """Entraîne le modèle sur une époque."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


def evaluate(model, dataloader, criterion, device):
    """Évalue le modèle sur un jeu de données."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, f1, precision, recall, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, gradient_clip=None, model_name="model"):
    """Boucle d'entraînement complète avec early stopping."""
    logger = MetricsLogger()
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    from utils.config import MODELS_DIR, PLOTS_DIR, RESULTS_DIR
    
    print(f"\n{'='*60}")
    print(f"Entraînement : {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )
        val_loss, val_acc, val_f1, _, _, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.log(train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, current_lr)
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{model_name}_best.pt'))
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping à l'époque {epoch} (meilleur F1 val: {best_val_f1:.4f} à l'époque {best_epoch})")
            break
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'{model_name}_best.pt'), weights_only=True))
    
    logger.save(os.path.join(RESULTS_DIR, f'{model_name}_history.json'))
    return logger, best_val_f1, best_epoch


def plot_learning_curves(logger, model_name, save_dir):
    """Trace les courbes d'apprentissage (loss et métriques)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(logger.history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, logger.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, logger.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoque')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Courbe de Loss - {model_name}')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, logger.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, logger.history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoque')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Courbe d\'Accuracy - {model_name}')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].plot(epochs, logger.history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, logger.history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoque')
    axes[2].set_ylabel('F1-Score (macro)')
    axes[2].set_title(f'Courbe de F1-Score - {model_name}')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes d'apprentissage sauvegardées : {model_name}_learning_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """Trace la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Matrice de Confusion - {model_name}')
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_xlabel('Classe predite')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Matrice de Confusion Normalisee - {model_name}')
    axes[1].set_ylabel('Vraie classe')
    axes[1].set_xlabel('Classe predite')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Matrice de confusion sauvegardee : {model_name}_confusion_matrix.png")


def plot_roc_curves(y_true, y_probs, n_classes, class_names, model_name, save_dir):
    """Trace les courbes ROC."""
    # One-hot encode les labels
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1
    
    y_probs = np.array(y_probs)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    aucs = []
    for i in range(min(n_classes, 10)):  # Limiter à 10 classes max pour la lisibilité
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        auc_val = roc_auc_score(y_onehot[:, i], y_probs[:, i])
        aucs.append(auc_val)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{class_names[i]} (AUC = {auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title(f'Courbe ROC - {model_name}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Courbes ROC sauvegardees : {model_name}_roc_curves.png")
    return aucs


def generate_classification_report(y_true, y_pred, class_names, model_name):
    """Génère un rapport de classification détaillé."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\nRapport de classification - {model_name}:")
    print(report)
    return report
