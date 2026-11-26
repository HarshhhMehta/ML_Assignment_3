import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.dataset import load_raw_cifar10, get_or_create_stratified_split, CIFAR10Dataset
from src.models.mlp import HighCapacityMLP

def main():
    torch.set_num_threads(8)
    device = 'cpu'
    seed = 26013
    
    print("=" * 70)
    print("AUDITING Q1(b) MEMORIZATION CHECKPOINT ON SUBSETS")
    print("=" * 70)
    
    # 1. Load data
    train_data_raw, train_labels_raw, test_data_raw, test_labels_raw = load_raw_cifar10()
    train_idx, val_idx = get_or_create_stratified_split(train_labels_raw, seed=seed)
    
    train_data = train_data_raw[train_idx]
    train_labels_clean = train_labels_raw[train_idx]
    val_data = train_data_raw[val_idx]
    val_labels_clean = train_labels_raw[val_idx]
    
    # Reconstruct exact noise partition
    rng = np.random.RandomState(seed)
    n_train = len(train_labels_clean)
    num_corrupt = int(n_train * 0.5)
    selected_corrupt_idx = rng.choice(n_train, size=num_corrupt, replace=False)
    
    assigned_noise = rng.randint(0, 10, size=num_corrupt)
    train_labels_corrupted = np.copy(train_labels_clean)
    train_labels_corrupted[selected_corrupt_idx] = assigned_noise
    
    # Analyze effective vs nominal corruption
    actually_changed_mask = (assigned_noise != train_labels_clean[selected_corrupt_idx])
    num_actually_changed = int(np.sum(actually_changed_mask))
    effective_corrupt_rate = num_actually_changed / n_train
    print(f"Nominal selected for corruption: {num_corrupt}/{n_train} ({num_corrupt/n_train*100:.1f}%)")
    print(f"Actually changed to different class: {num_actually_changed}/{n_train} ({effective_corrupt_rate*100:.2f}%)")
    print(f"Chanced back to true class: {num_corrupt - num_actually_changed}/{n_train} ({(num_corrupt - num_actually_changed)/n_train*100:.2f}%)")
    
    # Create mask for clean subset vs corrupted subset
    clean_train_mask = np.ones(n_train, dtype=bool)
    clean_train_mask[selected_corrupt_idx] = False
    
    # 2. Load model
    ckpt_path = 'checkpoints/q1b_memorization_model.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint {ckpt_path} not found!")
        return
        
    model = HighCapacityMLP(input_dim=3072, hidden_dim=512, num_classes=10).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Evaluation function
    def eval_subset(data, targets):
        dataset = CIFAR10Dataset(data, targets, normalize='standard')
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        return correct / total
        
    # Evaluate subsets
    acc_clean_unselected_train = eval_subset(train_data[clean_train_mask], train_labels_clean[clean_train_mask])
    acc_corrupt_noisy_target = eval_subset(train_data[selected_corrupt_idx], train_labels_corrupted[selected_corrupt_idx])
    acc_corrupt_true_target = eval_subset(train_data[selected_corrupt_idx], train_labels_clean[selected_corrupt_idx])
    acc_val_clean = eval_subset(val_data, val_labels_clean)
    
    results = {
        'n_train': n_train,
        'nominal_noise_rate': 0.50,
        'effective_noise_rate': float(effective_corrupt_rate),
        'unselected_clean_subset_acc': float(acc_clean_unselected_train),
        'corrupted_subset_acc_vs_noisy_target': float(acc_corrupt_noisy_target),
        'corrupted_subset_acc_vs_true_clean_target': float(acc_corrupt_true_target),
        'clean_val_acc': float(acc_val_clean),
        'discrepancy_gap_noisy_vs_val': float(acc_corrupt_noisy_target - acc_val_clean),
        'true_generalization_gap_clean_train_vs_val': float(acc_clean_unselected_train - acc_val_clean)
    }
    
    print("\n--- DETAILED SUBSET EVALUATION RESULTS ---")
    print(f"1. Unselected Clean Train Subset Acc (N={np.sum(clean_train_mask)}): {acc_clean_unselected_train*100:.2f}%")
    print(f"2. Corrupted Train Subset Acc vs. Noisy Target (Memorization) (N={num_corrupt}): {acc_corrupt_noisy_target*100:.2f}%")
    print(f"3. Corrupted Train Subset Acc vs. True Ground-Truth Label: {acc_corrupt_true_target*100:.2f}%")
    print(f"4. Clean Validation Set Acc (N={len(val_labels_clean)}): {acc_val_clean*100:.2f}%")
    print(f"5. True In-Distribution Gen Gap (Clean Train - Clean Val): {(acc_clean_unselected_train - acc_val_clean)*100:.2f}%")
    print(f"6. Cross-Distribution Gap (Noisy Target Acc - Clean Val): {(acc_corrupt_noisy_target - acc_val_clean)*100:.2f}%")
    
    with open('experiments/q1b_subset_audit.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved Q1(b) subset audit to experiments/q1b_subset_audit.json")

if __name__ == '__main__':
    main()
