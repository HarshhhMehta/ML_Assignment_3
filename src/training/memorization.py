import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import CIFAR10Dataset, load_raw_cifar10, get_or_create_stratified_split
from src.models.mlp import HighCapacityMLP

def run_memorization_experiment(epochs=35, batch_size=128, lr=1e-3, corruption_ratio=0.5,
                                device='cpu', seed=26013,
                                log_path='experiments/q1b_memorization_history.json',
                                checkpoint_path='checkpoints/q1b_memorization_model.pt'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data_raw, train_labels_raw, test_data_raw, test_labels_raw = load_raw_cifar10()
    train_idx, val_idx = get_or_create_stratified_split(train_labels_raw, seed=seed)
    
    train_data = train_data_raw[train_idx]
    train_labels_clean = train_labels_raw[train_idx]
    
    val_data = train_data_raw[val_idx]
    val_labels_clean = train_labels_raw[val_idx]
    
    # Corrupt exactly 50% of the training labels uniformly at random
    rng = np.random.RandomState(seed)
    n_train = len(train_labels_clean)
    num_corrupt = int(n_train * corruption_ratio)
    corrupt_indices = rng.choice(n_train, size=num_corrupt, replace=False)
    
    train_labels_corrupted = np.copy(train_labels_clean)
    train_labels_corrupted[corrupt_indices] = rng.randint(0, 10, size=num_corrupt)
    
    print(f"Q1(b) Memorization Setup: {n_train} train images, {num_corrupt} corrupted ({corruption_ratio*100:.0f}% noise).")
    print(f"Validation set: {len(val_labels_clean)} images with 100% clean labels.")
    
    train_dataset = CIFAR10Dataset(train_data, train_labels_clean, normalize='standard', corrupted_labels=train_labels_corrupted)
    val_dataset = CIFAR10Dataset(val_data, val_labels_clean, normalize='standard')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = HighCapacityMLP(input_dim=3072, hidden_dim=512, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],          # Accuracy on corrupted training targets
        'val_loss': [],
        'val_acc': [],            # Accuracy on clean validation targets
        'generalization_gap': [], # train_acc - val_acc
        'epoch_times': []
    }
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print("Beginning memorization trajectory tracking...")
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_time = time.time() - start_t
        train_loss = running_loss / total_train
        train_acc = train_correct / total_train
        
        # Validation on clean labels
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        val_loss = val_running_loss / total_val
        val_acc = val_correct / total_val
        gen_gap = train_acc - val_acc
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['generalization_gap'].append(gen_gap)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Train Acc (Noisy): {train_acc*100:.2f}% | Val Acc (Clean): {val_acc*100:.2f}% | Gen Gap: {gen_gap*100:.2f}%", flush=True)
        
    torch.save(model.state_dict(), checkpoint_path)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Memorization experiment complete. Log saved to {log_path}.")
    return history
