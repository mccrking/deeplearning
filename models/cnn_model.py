"""
Modèle CNN (Réseau de Neurones Convolutif)
Projet Deep Learning - Mehdi Chmiti - 4IAD G3
Dataset : CIFAR-10 (classification d'images 10 classes)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.metrics_logger import (
    train_model, evaluate, plot_learning_curves,
    plot_confusion_matrix, plot_roc_curves, generate_classification_report
)


class CNN(nn.Module):
    """
    CNN personnalisé pour la classification d'images.
    Architecture : 4 blocs convolutifs + couches denses.
    Conv2d → BatchNorm → ReLU → MaxPool → ... → AdaptiveAvgPool → Dense
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super(CNN, self).__init__()
        
        # Bloc 1 : 3 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bloc 2 : 32 → 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bloc 3 : 64 → 128
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Bloc 4 : 128 → 256
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifieur
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


class CNNTransferLearning(nn.Module):
    """
    CNN avec transfert d'apprentissage (ResNet18 pré-entraîné).
    Feature extraction : geler les couches convolutives.
    Fine-tuning : dégeler les dernières couches.
    """
    def __init__(self, num_classes=10, fine_tune=False):
        super(CNNTransferLearning, self).__init__()
        
        self.backbone = torchvision.models.resnet18(weights=None)
        
        # Geler les couches convolutives
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remplacer la couche de classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_cifar10_loaders(batch_size=64, augmentation=True, subset_ratio=1.0):
    """Charge le dataset CIFAR-10 avec prétraitement et augmentation."""
    
    # Transformations pour l'entraînement (avec augmentation)
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2470, 0.2435, 0.2616])
        ])
    
    # Transformations pour val/test (sans augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2470, 0.2435, 0.2616])
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    
    # Sous-ensemble si demandé (pour accélérer l'entraînement)
    if subset_ratio < 1.0:
        train_size = int(len(train_dataset) * subset_ratio)
        test_size = int(len(test_dataset) * subset_ratio)
        
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    # Split train/val (85/15 du train)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Appliquer le test_transform au validation set
    val_dataset_base = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=False, transform=test_transform
    )
    val_indices = val_subset.indices
    val_dataset = Subset(val_dataset_base, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def run_cnn_experiment():
    """Exécute l'expérimentation complète du CNN."""
    print("\n" + "="*70)
    print("  EXPERIMENTATION CNN - CIFAR-10")
    print("="*70)
    
    CIFAR10_CLASSES = ['Avion', 'Auto', 'Oiseau', 'Chat', 'Cerf',
                       'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']
    
    # === 1. Chargement des données ===
    print("\nChargement de CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=CNN_CONFIG['batch_size'],
        augmentation=CNN_CONFIG['augmentation'],
        subset_ratio=0.3  # 30% du dataset pour accélérer
    )
    
    print(f"  Train : {len(train_loader.dataset)} échantillons")
    print(f"  Val   : {len(val_loader.dataset)} échantillons")
    print(f"  Test  : {len(test_loader.dataset)} échantillons")
    
    # === 2. Définition du modèle ===
    set_seed()
    model = CNN(num_classes=10, dropout=CNN_CONFIG['dropout']).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Modèle CNN :")
    print(f"  Paramètres totaux : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}")
    
    # === 3. Entraînement ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CNN_CONFIG['learning_rate'],
        weight_decay=CNN_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
    )
    
    logger, best_f1, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        DEVICE, CNN_CONFIG['epochs'], CNN_CONFIG['patience'],
        model_name="CNN"
    )
    
    # === 4. Évaluation finale ===
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n{'='*50}")
    print(f"  RÉSULTATS FINAUX CNN (Test Set)")
    print(f"{'='*50}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_prec:.4f}")
    print(f"  Recall    : {test_rec:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    
    # === 5. Visualisations ===
    plot_learning_curves(logger, "CNN", PLOTS_DIR)
    plot_confusion_matrix(test_labels, test_preds, CIFAR10_CLASSES, "CNN", PLOTS_DIR)
    
    # ROC pour les 10 classes
    try:
        aucs = plot_roc_curves(test_labels, test_probs, 10, CIFAR10_CLASSES, "CNN", PLOTS_DIR)
    except Exception as e:
        print(f"  ROC curves erreur : {e}")
        aucs = []
    
    report = generate_classification_report(test_labels, test_preds, CIFAR10_CLASSES, "CNN")
    
    # Sauvegarde des résultats
    import json
    results = {
        'model': 'CNN',
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_f1),
        'best_epoch': int(best_epoch),
        'total_params': int(total_params),
        'augmentation': CNN_CONFIG['augmentation'],
    }
    
    with open(os.path.join(RESULTS_DIR, 'CNN_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_cnn_experiment()
