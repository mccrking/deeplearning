
import os
import sys
import json
import time

from utils.config import set_seed, DEVICE, RESULTS_DIR, PLOTS_DIR

def main():
    print("="*70)
    print(f"  Device : {DEVICE}")
    print("="*70)
    
    all_results = {}
    start_time = time.time()
    
    # === 1. MLP - Breast Cancer Wisconsin ===
    print("\n\n" + "▶"*30 + " PHASE 1 : MLP " + "▶"*30)
    try:
        from models.mlp_model import run_mlp_experiment
        all_results['MLP'] = run_mlp_experiment()
    except Exception as e:
        print(f"Erreur MLP : {e}")
        import traceback
        traceback.print_exc()
    
    # === 2. CNN - CIFAR-10 ===
    print("\n\n" + "▶"*30 + " PHASE 2 : CNN " + "▶"*30)
    try:
        from models.cnn_model import run_cnn_experiment
        all_results['CNN'] = run_cnn_experiment()
    except Exception as e:
        print(f"Erreur CNN : {e}")
        import traceback
        traceback.print_exc()
    
    # === 3. RNN / LSTM / GRU - Signaux synthétiques ===
    print("\n\n" + "▶"*30 + " PHASE 3 : RNN/LSTM/GRU " + "▶"*30)
    try:
        from models.rnn_models import run_all_rnn_experiments
        rnn_results = run_all_rnn_experiments()
        all_results.update(rnn_results)
    except Exception as e:
        print(f"Erreur RNN : {e}")
        import traceback
        traceback.print_exc()
    
    # === 4. Architecture Hybride CNN+LSTM ===
    print("\n\n" + "▶"*30 + " PHASE 4 : HYBRIDE " + "▶"*30)
    try:
        from models.hybrid_model import run_hybrid_experiment
        all_results['CNN_LSTM'] = run_hybrid_experiment()
    except Exception as e:
        print(f"Erreur Hybride : {e}")
        import traceback
        traceback.print_exc()
    
    # === Résumé global ===
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  RÉSUMÉ GLOBAL DES EXPÉRIMENTATIONS")
    print(f"{'='*70}")
    print(f"  Temps total : {elapsed/60:.1f} minutes")
    print(f"\n  {'Modèle':<12} {'Accuracy':>10} {'F1-Score':>10} {'Params':>12}")
    print(f"  {'-'*46}")
    
    for name, res in all_results.items():
        acc = res.get('test_accuracy', 0)
        f1 = res.get('test_f1', 0)
        params = res.get('total_params', 0)
        print(f"  {name:<12} {acc:>10.4f} {f1:>10.4f} {params:>12,}")
    
    # Sauvegarde du résumé
    with open(os.path.join(RESULTS_DIR, 'all_results_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Graphique comparatif global
    plot_global_comparison(all_results)
    
    print(f"\n  Tous les résultats sauvegardés dans : {RESULTS_DIR}")
    print(f"  Toutes les visualisations sauvegardées dans : {PLOTS_DIR}")
    
    return all_results


def plot_global_comparison(results):
    """Trace un graphique comparatif de tous les modèles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    models = list(results.keys())
    if not models:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy et F1
    x = np.arange(len(models))
    width = 0.35
    
    accuracies = [results[m].get('test_accuracy', 0) for m in models]
    f1_scores = [results[m].get('test_f1', 0) for m in models]
    
    bars1 = axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    bars2 = axes[0].bar(x + width/2, f1_scores, width, label='F1-Score', color='#DD8452', alpha=0.85)
    
    axes[0].set_xlabel('Modele')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Comparaison globale des performances')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend(loc='best')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    # Nombre de paramètres
    params = [results[m].get('total_params', 0) for m in models]
    bars = axes[1].bar(models, params, color='#55A868', alpha=0.85)
    axes[1].set_xlabel('Modele')
    axes[1].set_ylabel('Nombre de parametres')
    axes[1].set_title('Complexite des modeles (parametres)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'global_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparaison globale sauvegardée : global_comparison.png")


if __name__ == "__main__":
    results = main()
