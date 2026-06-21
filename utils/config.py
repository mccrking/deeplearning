

import torch
import numpy as np
import random
import os

# === Reproductibilité ===
SEED = 42
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# === Device ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé : {DEVICE}")

# === Chemins ===
PROJECT_DIR = "./"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# === Hyperparamètres par architecture ===
MLP_CONFIG = {
    'hidden_sizes': [128, 64, 32],
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 50,
    'weight_decay': 1e-4,
    'patience': 10,
}

CNN_CONFIG = {
    'num_blocks': 4,
    'base_filters': 32,
    'kernel_size': 3,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 30,
    'weight_decay': 1e-4,
    'patience': 7,
    'augmentation': True,
}

RNN_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'bidirectional': True,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'epochs': 20,
    'weight_decay': 1e-4,
    'patience': 6,
    'gradient_clip': 1.0,
}

HYBRID_CONFIG = {
    'cnn_filters': [32, 64, 128],
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'dropout': 0.3,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'epochs': 20,
    'weight_decay': 1e-4,
    'patience': 6,
    'gradient_clip': 1.0,
}
