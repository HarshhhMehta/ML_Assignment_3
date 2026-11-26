import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from src.data.dataset import get_dataloaders
from src.models.mlp import StandardMLP, PatchLocallyConnectedMLP, count_parameters
from src.training.trainer import train_classifier, evaluate_classifier
from src.training.memorization import run_memorization_experiment
from src.utils.registry import log_experiment
from src.utils.plotting import plot_q1a_curves, plot_q1b_memorization

def main():
    torch.set_num_threads(8)
    device = 'cpu'
    seed = 26013
    epochs = 20
    batch_size = 64
    
    print("=" * 70)
    print("STARTING Q1 EXPERIMENTS: MLP LIMITATIONS AND MEMORIZATION")
    print("=" * 70)
    
    # Dataloaders for Q1(a)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, seed=seed)
    
    # -------------------------------------------------------------
    # Q1(a) Part 1: Standard MLP (Global Receptive Field)
    # -------------------------------------------------------------
    std_ckpt = 'checkpoints/q1a_standard_mlp.pt'
    std_log = 'experiments/q1a_standard_mlp_log.json'
    
    if os.path.exists(std_ckpt) and os.path.exists(std_log):
        print(f"[Q1(a)] Found existing Standard MLP checkpoint: {std_ckpt}. Loading...", flush=True)
        with open(std_log, 'r') as f:
            std_history = json.load(f)
        std_mlp = StandardMLP(input_dim=3072, hidden_dim=36, num_classes=10)
        std_mlp.load_state_dict(torch.load(std_ckpt))
        std_params = count_parameters(std_mlp)
    else:
        print("\n[Q1(a)] Training Standard MLP (3072 -> 36 -> 36 -> 10)...", flush=True)
        std_mlp = StandardMLP(input_dim=3072, hidden_dim=36, num_classes=10)
        std_params = count_parameters(std_mlp)
        print(f"Standard MLP Parameter Count: {std_params:,}", flush=True)
        std_history = train_classifier(
            std_mlp, train_loader, val_loader,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=std_ckpt, log_path=std_log, seed=seed
        )
        std_mlp.load_state_dict(torch.load(std_ckpt))
        std_test_loss, std_test_acc = evaluate_classifier(std_mlp, test_loader, device=device)
        print(f">> Standard MLP Official Test Accuracy: {std_test_acc*100:.2f}% (Test Loss: {std_test_loss:.4f})", flush=True)
        log_experiment(
            exp_id='Q1a_Standard_MLP',
            question='Q1(a)',
            model_name='StandardMLP',
            param_count=std_params,
            config={'epochs': epochs, 'batch_size': batch_size, 'lr': 1e-3, 'seed': seed},
            val_metric={'best_val_acc': std_history['best_val_acc'], 'best_epoch': std_history['best_epoch']},
            test_metric={'test_acc': std_test_acc, 'test_loss': std_test_loss},
            checkpoint_path=std_ckpt,
            notes='Baseline standard MLP with global connectivity.'
        )

    # -------------------------------------------------------------
    # Q1(a) Part 2: Patch Locally Connected MLP (Local Receptive Field)
    # -------------------------------------------------------------
    patch_ckpt = 'checkpoints/q1a_patch_lcn.pt'
    patch_log = 'experiments/q1a_patch_lcn_log.json'
    
    if os.path.exists(patch_ckpt) and os.path.exists(patch_log):
        print(f"[Q1(a)] Found existing Patch-LCN checkpoint: {patch_ckpt}. Loading...", flush=True)
        with open(patch_log, 'r') as f:
            patch_history = json.load(f)
        patch_lcn = PatchLocallyConnectedMLP(patch_size=8, hidden_per_patch=35, num_classes=10)
        patch_lcn.load_state_dict(torch.load(patch_ckpt))
        patch_params = count_parameters(patch_lcn)
    else:
        print("\n[Q1(a)] Training Patch-Locally Connected MLP (16 patches x 8x8, NO weight sharing)...", flush=True)
        patch_lcn = PatchLocallyConnectedMLP(patch_size=8, hidden_per_patch=35, num_classes=10)
        patch_params = count_parameters(patch_lcn)
        print(f"Patch-LCN Parameter Count: {patch_params:,}", flush=True)
        patch_history = train_classifier(
            patch_lcn, train_loader, val_loader,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=patch_ckpt, log_path=patch_log, seed=seed
        )
        patch_lcn.load_state_dict(torch.load(patch_ckpt))
        patch_test_loss, patch_test_acc = evaluate_classifier(patch_lcn, test_loader, device=device)
        print(f">> Patch-LCN Official Test Accuracy: {patch_test_acc*100:.2f}% (Test Loss: {patch_test_loss:.4f})", flush=True)
        log_experiment(
            exp_id='Q1a_Patch_LCN',
            question='Q1(a)',
            model_name='PatchLocallyConnectedMLP',
            param_count=patch_params,
            config={'epochs': epochs, 'batch_size': batch_size, 'lr': 1e-3, 'seed': seed},
            val_metric={'best_val_acc': patch_history['best_val_acc'], 'best_epoch': patch_history['best_epoch']},
            test_metric={'test_acc': patch_test_acc, 'test_loss': patch_test_loss},
            checkpoint_path=patch_ckpt,
            notes='Locally connected model isolating spatial locality from weight sharing.'
        )
    
    # Generate Q1(a) Publication Figures
    plot_q1a_curves(std_history, patch_history, save_path='figures/q1a_learning_curves.pdf')
    
    # -------------------------------------------------------------
    # Q1(b): Memorization Experiment Under 50% Label Noise
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Q1(b)] Running High-Capacity MLP Memorization Dynamics Trajectory...")
    print("=" * 70)
    
    mem_history = run_memorization_experiment(
        epochs=25, batch_size=128, lr=1e-3, corruption_ratio=0.5,
        device=device, seed=seed,
        log_path='experiments/q1b_memorization_history.json',
        checkpoint_path='checkpoints/q1b_memorization_model.pt'
    )
    
    plot_q1b_memorization(mem_history, save_path='figures/q1b_memorization_trajectory.pdf')
    
    # Identify key transition epochs
    val_accs = np.array(mem_history['val_acc'])
    peak_epoch = int(np.argmax(val_accs)) + 1
    peak_val_acc = float(np.max(val_accs))
    final_val_acc = float(val_accs[-1])
    final_train_acc = float(mem_history['train_acc'][-1])
    final_gen_gap = float(mem_history['generalization_gap'][-1])
    
    print("\n" + "-" * 50)
    print("Q1(b) TRAJECTORY ANALYSIS RESULTS:")
    print(f"Peak Validation Accuracy: {peak_val_acc*100:.2f}% achieved at Epoch {peak_epoch}")
    print(f"Final Epoch 30 Train Acc (Noisy): {final_train_acc*100:.2f}%")
    print(f"Final Epoch 30 Val Acc (Clean): {final_val_acc*100:.2f}%")
    print(f"Final Generalization Gap: {final_gen_gap*100:.2f}%")
    print("-" * 50)
    
    log_experiment(
        exp_id='Q1b_Memorization',
        question='Q1(b)',
        model_name='HighCapacityMLP',
        param_count=1841162,
        config={'epochs': 30, 'batch_size': 128, 'noise_ratio': 0.5, 'lr': 1e-3, 'seed': seed},
        val_metric={'peak_val_acc': peak_val_acc, 'peak_epoch': peak_epoch, 'final_val_acc': final_val_acc},
        test_metric={'final_train_acc_noisy': final_train_acc, 'final_gen_gap': final_gen_gap},
        checkpoint_path='checkpoints/q1b_memorization_model.pt',
        notes='Tracks early generalizable feature learning vs late brute-force label memorization.'
    )
    print("\nQ1 EXPERIMENTS FULLY COMPLETE!")

if __name__ == '__main__':
    main()
