# Structure du projet Deep Learning — CIFAR-10

```
deep_learning_project/
│
├── main.py                          # Script principal : entraînement + évaluation
├── STRUCTURE.md                     # Structure détaillée
│
├── utils/                           # Utilitaires
│   ├── config.py                    # Configuration globale (hyperparamètres, chemins)
│   └── metrics_logger.py            # Logging des métriques (JSON + W&B style)
│
├── models/                          # Définitions + poids des modèles
│   ├── mlp_model.py                 # MLP (baseline)
│   ├── cnn_model.py                 # CNN convolutionnel
│   ├── rnn_models.py                # RNN, LSTM, GRU
│   ├── hybrid_model.py              # CNN-LSTM hybride
│   ├── MLP_best.pt                  # Poids entraînés
│   ├── CNN_best.pt
│   ├── RNN_best.pt
│   ├── LSTM_best.pt
│   ├── GRU_best.pt
│   └── CNN_LSTM_best.pt
│
├── data/                            # Dataset CIFAR-10 (auto-téléchargé)
│   ├── cifar-10-python.tar.gz       # Archive (à supprimer après extraction)
│   └── cifar-10-batches-py/         # 5 batches train + 1 test + meta
│
├── results/                         # Résultats d'évaluation
│   ├── all_results_summary.json     # Résumé comparatif de tous les modèles
│   ├── {model}_results.json         # Métriques finales (accuracy, F1, etc.)
│   ├── {model}_history.json         # Historique epoch par epoch
│   └── plots/                       # Graphiques générés
│       ├── global_comparison.png    # Comparaison globale
│       ├── {model}_learning_curves.png   # Courbes loss/accuracy
│       ├── {model}_confusion_matrix.png  # Matrices de confusion
│       ├── {model}_roc_curves.png        # Courbes ROC multiclasse
│       ├── MLP_feature_importance.png    # Importance des features (MLP)
│       └── RNN_comparison.png            # Comparaison RNN vs variants
│               
```

## Modèles implémentés

| Modèle       | Type              | But                              |
|--------------|-------------------|----------------------------------|
| MLP          | Baseline dense    | Référence simple                 |
| CNN          | Convolutionnel    | Adapté aux images                |
| RNN          | Récurrent simple  | Test séquentiel sur pixels       |
| LSTM         | Récurrent gated   | Mémoire long terme               |
| GRU          | Récurrent léger   | Variante optimisée du LSTM       |
| CNN-LSTM     | Hybride           | Combine CNN (features) + LSTM    |

## Comment exécuter

```bash
# 1. Installer les dépendances
pip install torch torchvision scikit-learn matplotlib numpy

# 2. Lancer l'entraînement de tous les modèles
python main.py

# 3. Les résultats sont générés dans results/ et models/
```
