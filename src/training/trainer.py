import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

def train_classifier(model, train_loader, val_loader,
                     epochs=25, lr=1e-3, weight_decay=1e-4,
                     device='cpu', checkpoint_path='checkpoints/best_model.pt',
                     log_path='experiments/train_log.json',
                     seed=26013):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': epochs,
        'best_val_acc': 0.0,
        'best_epoch': 0,
        'epoch_times': []
    }
    
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"Starting classifier training for {epochs} epochs on {device}...")
    
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()
        epoch_time = time.time() - start_t
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation evaluation
        val_loss, val_acc = evaluate_classifier(model, val_loader, device=device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_path)
            
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Training finished. Best Val Acc: {best_val_acc*100:.2f}% at epoch {history['best_epoch']}.")
    return history

def evaluate_classifier(model, loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    return running_loss / total, correct / total

def train_autoencoder(model, train_loader, val_loader,
                      epochs=25, lr=1e-3, weight_decay=1e-5,
                      device='cpu', checkpoint_path='checkpoints/best_ae.pt',
                      log_path='experiments/ae_train_log.json',
                      seed=26013, is_denoising=False, noise_sigma=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_mse': [],
        'val_mse': [],
        'epochs': epochs,
        'best_val_mse': float('inf'),
        'best_epoch': 0,
        'epoch_times': []
    }
    
    best_val_mse = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    mode_str = "Denoising Autoencoder" if is_denoising else "Clean Autoencoder"
    print(f"Starting {mode_str} training for {epochs} epochs on {device}...")
    
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss = 0.0
        total = 0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            if is_denoising:
                # Add Gaussian noise to input
                noise = torch.randn_like(images) * noise_sigma
                corrupted_images = torch.clamp(images + noise, 0.0, 1.0)
                inputs = corrupted_images
            else:
                inputs = images
                
            targets = images # Target is always the clean image
            
            optimizer.zero_grad()
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            
        scheduler.step()
        epoch_time = time.time() - start_t
        train_mse = running_loss / total
        
        # Validation MSE (on clean inputs if clean AE, or noisy inputs if denoising AE)
        val_mse = evaluate_autoencoder(model, val_loader, device=device, is_denoising=is_denoising, noise_sigma=noise_sigma)
        
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            history['best_val_mse'] = best_val_mse
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_path)
            
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Autoencoder training finished. Best Val MSE: {best_val_mse:.6f} at epoch {history['best_epoch']}.")
    return history

def evaluate_autoencoder(model, loader, device='cpu', is_denoising=False, noise_sigma=0.2):
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if is_denoising:
                noise = torch.randn_like(images) * noise_sigma
                inputs = torch.clamp(images + noise, 0.0, 1.0)
            else:
                inputs = images
            targets = images
            reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            
    return running_loss / total
