import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from src.data.dataset import get_dataloaders
from src.data.transforms import PixelPermutationTransform, PatchPermutationTransform
from src.models.mlp import StandardMLP, count_parameters as count_mlp_params
from src.models.cnn import FairCNN, count_parameters as count_cnn_params
from src.training.trainer import train_classifier, evaluate_classifier
from src.evaluation.metrics import compute_confidence_and_calibration
from src.utils.registry import log_experiment
from src.utils.plotting import plot_q2_comparisons, plot_q2c_confidence_histograms

def main():
    torch.set_num_threads(8)
    device = 'cpu'
    seed = 26013
    epochs = 20
    batch_size = 64
    
    print("=" * 70, flush=True)
    print("STARTING Q2 EXPERIMENTS: CNN VS MLP, SPATIAL STRUCTURE & CALIBRATION", flush=True)
    print("=" * 70, flush=True)
    
    # -----------------------------------------------------------------
    # Transforms (Deterministic with seed=26013)
    # -----------------------------------------------------------------
    pixel_transform = PixelPermutationTransform(seed=seed, channel_aligned=True)
    patch_transform = PatchPermutationTransform(patch_size=8, seed=seed)
    
    # Dataloaders
    clean_train, clean_val, clean_test = get_dataloaders(batch_size=batch_size, seed=seed)
    pix_train, pix_val, pix_test = get_dataloaders(batch_size=batch_size, transform=pixel_transform, seed=seed)
    patch_train, patch_val, patch_test = get_dataloaders(batch_size=batch_size, transform=patch_transform, seed=seed)
    
    results_summary = {
        'MLP': {},
        'CNN': {}
    }
    
    # -----------------------------------------------------------------
    # Q2(a) Part 1: Reused Clean MLP from Q1(a)
    # -----------------------------------------------------------------
    mlp_clean_ckpt = 'checkpoints/q1a_standard_mlp.pt'
    mlp_clean = StandardMLP(input_dim=3072, hidden_dim=36, num_classes=10)
    mlp_clean.load_state_dict(torch.load(mlp_clean_ckpt, map_location=device))
    mlp_params = count_mlp_params(mlp_clean)
    _, mlp_clean_test_acc = evaluate_classifier(mlp_clean, clean_test, device=device)
    results_summary['MLP']['Clean'] = mlp_clean_test_acc
    print(f">> [Reused from Q1(a)] Clean MLP Test Acc: {mlp_clean_test_acc*100:.2f}% (Params: {mlp_params:,})", flush=True)
    
    # -----------------------------------------------------------------
    # Q2(a) Part 2: Clean Fair CNN
    # -----------------------------------------------------------------
    cnn_clean = FairCNN(num_classes=10)
    cnn_params = count_cnn_params(cnn_clean)
    print(f">> Fair CNN Parameter Count: {cnn_params:,} (Parity vs MLP: {abs(cnn_params - mlp_params):,} diff, {abs(cnn_params - mlp_params)/mlp_params*100:.2f}%)", flush=True)
    
    cnn_clean_ckpt = 'checkpoints/q2a_fair_cnn_clean.pt'
    cnn_clean_log = 'experiments/q2a_fair_cnn_clean_log.json'
    
    if os.path.exists(cnn_clean_ckpt) and os.path.exists(cnn_clean_log):
        print(f">> [Reused] Loading existing Clean CNN checkpoint: {cnn_clean_ckpt}", flush=True)
        cnn_clean.load_state_dict(torch.load(cnn_clean_ckpt, map_location=device))
    else:
        print("\n[Q2(a)] Training Fair CNN on Clean CIFAR-10...", flush=True)
        train_classifier(
            cnn_clean, clean_train, clean_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=cnn_clean_ckpt, log_path=cnn_clean_log, seed=seed
        )
        cnn_clean.load_state_dict(torch.load(cnn_clean_ckpt, map_location=device))
        
    _, cnn_clean_test_acc = evaluate_classifier(cnn_clean, clean_test, device=device)
    results_summary['CNN']['Clean'] = cnn_clean_test_acc
    print(f">> Clean CNN Official Test Accuracy: {cnn_clean_test_acc*100:.2f}%", flush=True)
    
    # -----------------------------------------------------------------
    # Q2(b): Pixel-Permuted CIFAR-10 (MLP vs CNN)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q2(b)] SPATIAL DISTURBANCE: PIXEL-PERMUTED CIFAR-10", flush=True)
    print("=" * 70, flush=True)
    
    # 1. MLP on Pixel-Permuted Data
    mlp_pix = StandardMLP(input_dim=3072, hidden_dim=36, num_classes=10)
    mlp_pix_ckpt = 'checkpoints/q2b_mlp_pixel_permuted.pt'
    mlp_pix_log = 'experiments/q2b_mlp_pixel_permuted_log.json'
    
    if os.path.exists(mlp_pix_ckpt) and os.path.exists(mlp_pix_log):
        print(f">> [Reused] Loading existing Pixel-Permuted MLP checkpoint: {mlp_pix_ckpt}", flush=True)
        mlp_pix.load_state_dict(torch.load(mlp_pix_ckpt, map_location=device))
    else:
        print("Training MLP on Pixel-Permuted CIFAR-10...", flush=True)
        train_classifier(
            mlp_pix, pix_train, pix_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=mlp_pix_ckpt, log_path=mlp_pix_log, seed=seed
        )
        mlp_pix.load_state_dict(torch.load(mlp_pix_ckpt, map_location=device))
        
    _, mlp_pix_test_acc = evaluate_classifier(mlp_pix, pix_test, device=device)
    results_summary['MLP']['Pixel Permuted'] = mlp_pix_test_acc
    print(f">> Pixel-Permuted MLP Official Test Accuracy: {mlp_pix_test_acc*100:.2f}% (Clean: {results_summary['MLP']['Clean']*100:.2f}%)", flush=True)
    
    # 2. CNN on Pixel-Permuted Data
    cnn_pix = FairCNN(num_classes=10)
    cnn_pix_ckpt = 'checkpoints/q2b_cnn_pixel_permuted.pt'
    cnn_pix_log = 'experiments/q2b_cnn_pixel_permuted_log.json'
    
    if os.path.exists(cnn_pix_ckpt) and os.path.exists(cnn_pix_log):
        print(f">> Loading existing Pixel-Permuted CNN checkpoint: {cnn_pix_ckpt}", flush=True)
        cnn_pix.load_state_dict(torch.load(cnn_pix_ckpt, map_location=device))
    else:
        print("Training Fair CNN on Pixel-Permuted CIFAR-10...", flush=True)
        train_classifier(
            cnn_pix, pix_train, pix_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=cnn_pix_ckpt, log_path=cnn_pix_log, seed=seed
        )
        cnn_pix.load_state_dict(torch.load(cnn_pix_ckpt, map_location=device))
        
    _, cnn_pix_test_acc = evaluate_classifier(cnn_pix, pix_test, device=device)
    results_summary['CNN']['Pixel Permuted'] = cnn_pix_test_acc
    print(f">> Pixel-Permuted CNN Official Test Accuracy: {cnn_pix_test_acc*100:.2f}% (Clean: {results_summary['CNN']['Clean']*100:.2f}%)", flush=True)
    
    log_experiment(
        exp_id='Q2b_CNN_Pixel_Permuted',
        question='Q2(b)',
        model_name='FairCNN',
        param_count=cnn_params,
        config={'epochs': epochs, 'batch_size': batch_size, 'transform': 'PixelPermutation', 'seed': seed},
        val_metric={'test_acc': cnn_pix_test_acc},
        test_metric={'test_acc': cnn_pix_test_acc},
        checkpoint_path=cnn_pix_ckpt,
        notes='CNN trained and evaluated on pixel-permuted CIFAR-10.'
    )
    
    # -----------------------------------------------------------------
    # Q2(c): Third Distinguishing Condition (Patch-Permuted CIFAR-10)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q2(c)] THIRD CONDITION: PATCH-PERMUTED CIFAR-10 (8x8 BLOCKS)", flush=True)
    print("=" * 70, flush=True)
    
    # 1. MLP on Patch-Permuted Data
    mlp_patch = StandardMLP(input_dim=3072, hidden_dim=36, num_classes=10)
    mlp_patch_ckpt = 'checkpoints/q2c_mlp_patch_permuted.pt'
    mlp_patch_log = 'experiments/q2c_mlp_patch_permuted_log.json'
    
    if os.path.exists(mlp_patch_ckpt) and os.path.exists(mlp_patch_log):
        print(f">> Loading existing Patch-Permuted MLP checkpoint: {mlp_patch_ckpt}", flush=True)
        mlp_patch.load_state_dict(torch.load(mlp_patch_ckpt, map_location=device))
    else:
        print("Training MLP on Patch-Permuted CIFAR-10...", flush=True)
        train_classifier(
            mlp_patch, patch_train, patch_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=mlp_patch_ckpt, log_path=mlp_patch_log, seed=seed
        )
        mlp_patch.load_state_dict(torch.load(mlp_patch_ckpt, map_location=device))
        
    _, mlp_patch_test_acc = evaluate_classifier(mlp_patch, patch_test, device=device)
    results_summary['MLP']['Patch Permuted'] = mlp_patch_test_acc
    print(f">> Patch-Permuted MLP Official Test Accuracy: {mlp_patch_test_acc*100:.2f}% (Clean: {results_summary['MLP']['Clean']*100:.2f}%)", flush=True)
    
    log_experiment(
        exp_id='Q2c_MLP_Patch_Permuted',
        question='Q2(c)',
        model_name='StandardMLP',
        param_count=mlp_params,
        config={'epochs': epochs, 'batch_size': batch_size, 'transform': 'PatchPermutation', 'seed': seed},
        val_metric={'test_acc': mlp_patch_test_acc},
        test_metric={'test_acc': mlp_patch_test_acc},
        checkpoint_path=mlp_patch_ckpt,
        notes='MLP trained and evaluated on 8x8 patch-permuted CIFAR-10.'
    )
    
    # 2. CNN on Patch-Permuted Data
    cnn_patch = FairCNN(num_classes=10)
    cnn_patch_ckpt = 'checkpoints/q2c_cnn_patch_permuted.pt'
    cnn_patch_log = 'experiments/q2c_cnn_patch_permuted_log.json'
    
    if os.path.exists(cnn_patch_ckpt) and os.path.exists(cnn_patch_log):
        print(f">> Loading existing Patch-Permuted CNN checkpoint: {cnn_patch_ckpt}", flush=True)
        cnn_patch.load_state_dict(torch.load(cnn_patch_ckpt, map_location=device))
    else:
        print("Training Fair CNN on Patch-Permuted CIFAR-10...", flush=True)
        train_classifier(
            cnn_patch, patch_train, patch_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=cnn_patch_ckpt, log_path=cnn_patch_log, seed=seed
        )
        cnn_patch.load_state_dict(torch.load(cnn_patch_ckpt, map_location=device))
        
    _, cnn_patch_test_acc = evaluate_classifier(cnn_patch, patch_test, device=device)
    results_summary['CNN']['Patch Permuted'] = cnn_patch_test_acc
    print(f">> Patch-Permuted CNN Official Test Accuracy: {cnn_patch_test_acc*100:.2f}% (Clean: {results_summary['CNN']['Clean']*100:.2f}%)", flush=True)
    
    log_experiment(
        exp_id='Q2c_CNN_Patch_Permuted',
        question='Q2(c)',
        model_name='FairCNN',
        param_count=cnn_params,
        config={'epochs': epochs, 'batch_size': batch_size, 'transform': 'PatchPermutation', 'seed': seed},
        val_metric={'test_acc': cnn_patch_test_acc},
        test_metric={'test_acc': cnn_patch_test_acc},
        checkpoint_path=cnn_patch_ckpt,
        notes='CNN trained and evaluated on 8x8 patch-permuted CIFAR-10.'
    )
    
    # -----------------------------------------------------------------
    # Comprehensive Confidence and Calibration Analysis for BOTH Architectures
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("COMPREHENSIVE CONFIDENCE & CALIBRATION ANALYSIS (ECE) - BOTH MODELS", flush=True)
    print("=" * 70, flush=True)
    
    cnn_clean_conf = compute_confidence_and_calibration(cnn_clean, clean_test, device=device)
    cnn_patch_conf = compute_confidence_and_calibration(cnn_patch, patch_test, device=device)
    cnn_pix_conf   = compute_confidence_and_calibration(cnn_pix, pix_test, device=device)
    
    mlp_clean_conf = compute_confidence_and_calibration(mlp_clean, clean_test, device=device)
    mlp_patch_conf = compute_confidence_and_calibration(mlp_patch, patch_test, device=device)
    mlp_pix_conf   = compute_confidence_and_calibration(mlp_pix, pix_test, device=device)
    
    def strip_arrays(d):
        return {k: v for k, v in d.items() if k not in ['predictions', 'targets', 'confidences']}
    
    summary_report = {
        'results_summary': results_summary,
        'cnn_clean': strip_arrays(cnn_clean_conf),
        'cnn_patch': strip_arrays(cnn_patch_conf),
        'cnn_pixel': strip_arrays(cnn_pix_conf),
        'mlp_clean': strip_arrays(mlp_clean_conf),
        'mlp_patch': strip_arrays(mlp_patch_conf),
        'mlp_pixel': strip_arrays(mlp_pix_conf)
    }
    
    print("\n--- FINAL Q2 FACTORIAL COMPARISON MATRIX ---", flush=True)
    print(f"{'Architecture':<8} | {'Condition':<15} | {'Test Acc':<9} | {'Mean Conf':<9} | {'Conf (Corr)':<11} | {'Conf (Incorr)':<13} | {'ECE':<7}", flush=True)
    print("-" * 85, flush=True)
    for model_name, conf_map in [
        ('CNN', [('Clean', cnn_clean_conf), ('Patch-Permuted', cnn_patch_conf), ('Pixel-Permuted', cnn_pix_conf)]),
        ('MLP', [('Clean', mlp_clean_conf), ('Patch-Permuted', mlp_patch_conf), ('Pixel-Permuted', mlp_pix_conf)])
    ]:
        for cond_name, c_dict in conf_map:
            print(f"{model_name:<8} | {cond_name:<15} | {c_dict['accuracy']*100:6.2f}%  | {c_dict['overall_confidence']:7.3f}   | {c_dict['confidence_correct']:9.3f}   | {c_dict['confidence_incorrect']:11.3f}   | {c_dict['ece']:5.4f}", flush=True)
    print("-" * 85, flush=True)
    
    # Save figures
    plot_q2_comparisons(results_summary, save_path='figures/q2_spatial_disturbance.pdf')
    plot_q2c_confidence_histograms(cnn_clean_conf, cnn_pix_conf, save_path='figures/q2c_confidence_analysis.pdf')
    
    # Save full json summary
    os.makedirs('results', exist_ok=True)
    with open('results/q2_comparison_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    print("\nSaved comprehensive Q2 summary to results/q2_comparison_summary.json", flush=True)
    print("Q2 EXPERIMENTS FULLY COMPLETE!", flush=True)

if __name__ == '__main__':
    main()
