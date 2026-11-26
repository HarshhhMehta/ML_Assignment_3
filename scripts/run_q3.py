import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
from src.data.dataset import get_dataloaders
from src.models.autoencoder import ConvAutoencoder, get_ae_parameter_counts
from src.models.cnn import FairCNN
from src.training.trainer import train_autoencoder, evaluate_autoencoder
from src.utils.registry import log_experiment
from src.utils.plotting import plot_q3a_reconstruction_grid, plot_q3b_denoising_comparison, plot_q3c_tradeoff

def extract_features(model, loader, device='cpu'):
    model.eval()
    all_feats = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = model.encode(images)
            all_feats.append(feats.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)

def train_linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, num_classes=10, epochs=30, lr=1e-2, seed=26013):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = train_x.shape[1]
    
    probe = nn.Linear(in_dim, num_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    # Feature Standardization fitted strictly on training split ONLY
    mean = np.mean(train_x, axis=0, keepdims=True)
    std = np.std(train_x, axis=0, keepdims=True) + 1e-8
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std
    
    tx = torch.from_numpy(train_x).float()
    ty = torch.from_numpy(train_y).long()
    vx = torch.from_numpy(val_x).float()
    vy = torch.from_numpy(val_y).long()
    tex = torch.from_numpy(test_x).float()
    tey = torch.from_numpy(test_y).long()
    
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(epochs):
        probe.train()
        opt.zero_grad()
        out = probe(tx)
        loss = crit(out, ty)
        loss.backward()
        opt.step()
        
        probe.eval()
        with torch.no_grad():
            val_out = probe(vx)
            val_acc = (val_out.argmax(dim=1) == vy).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = probe.state_dict().copy()
                
    probe.load_state_dict(best_weights)
    probe.eval()
    with torch.no_grad():
        test_out = probe(tex)
        test_acc = (test_out.argmax(dim=1) == tey).float().mean().item()
        
    return best_val_acc, test_acc

def main():
    torch.set_num_threads(8)
    device = 'cpu'
    seed = 26013
    epochs = 20
    batch_size = 64
    noise_sigma = 0.2  # Gaussian noise N(0, 0.04)
    
    print("=" * 70, flush=True)
    print("STARTING Q3 EXPERIMENTS: CONVOLUTIONAL AUTOENCODERS, DENOISING & REPRESENTATIONS", flush=True)
    print("=" * 70, flush=True)
    
    # Dataloaders with unit normalization [0, 1] for Autoencoders
    unit_train, unit_val, unit_test = get_dataloaders(batch_size=batch_size, normalize='unit', seed=seed)
    
    # -----------------------------------------------------------------
    # Q3(a): Bottleneck Capacity Comparison (64, 256, 1024)
    # -----------------------------------------------------------------
    bottlenecks = [64, 256, 1024]
    ae_models = {}
    ae_val_mses = {}
    ae_params = {}
    
    for b in bottlenecks:
        print(f"\n[Q3(a)] Autoencoder with Bottleneck = {b}...", flush=True)
        ae = ConvAutoencoder(bottleneck_dim=b)
        p_info = get_ae_parameter_counts(ae)
        ae_params[b] = p_info
        print(f"Bottleneck {b} Dimensions: Enc={p_info['encoder_params']:,}, Dec={p_info['decoder_params']:,}, Total={p_info['total_params']:,}", flush=True)
        
        ckpt = f'checkpoints/q3a_clean_ae_b{b}.pt'
        log_p = f'experiments/q3a_clean_ae_b{b}_log.json'
        
        if os.path.exists(ckpt) and os.path.exists(log_p):
            print(f"Loading existing checkpoint: {ckpt}", flush=True)
            ae.load_state_dict(torch.load(ckpt, map_location=device))
            with open(log_p, 'r') as f:
                hist = json.load(f)
            ae_val_mses[b] = hist['best_val_mse']
        else:
            hist = train_autoencoder(
                ae, unit_train, unit_val,
                epochs=epochs, lr=1e-3, device=device,
                checkpoint_path=ckpt, log_path=log_p, seed=seed,
                is_denoising=False
            )
            ae.load_state_dict(torch.load(ckpt, map_location=device))
            ae_val_mses[b] = hist['best_val_mse']
            
        ae_models[b] = ae
        print(f">> Bottleneck {b}: Best Validation MSE = {ae_val_mses[b]:.6f}", flush=True)
        
    # Model Selection using VALIDATION MSE ONLY
    best_b = min(ae_val_mses, key=ae_val_mses.get)
    print("\n" + "-" * 60, flush=True)
    print(f">> FORMAL MODEL SELECTION (Validation MSE Only):", flush=True)
    print(f"   Selected Clean Bottleneck = {best_b} (Val MSE = {ae_val_mses[best_b]:.6f})", flush=True)
    print(f"   Selection rationale: Lowest reconstruction error across the 5,000-image validation split.", flush=True)
    print("-" * 60, flush=True)
    
    # Official Test Set Evaluation strictly after model selection
    ae_test_mses = {}
    for b in bottlenecks:
        test_mse = evaluate_autoencoder(ae_models[b], unit_test, device=device, is_denoising=False)
        ae_test_mses[b] = test_mse
        print(f">> Bottleneck {b} Official Test MSE: {test_mse:.6f}", flush=True)
        
        log_experiment(
            exp_id=f'Q3a_AE_b{b}',
            question='Q3(a)',
            model_name=f'ConvAutoencoder_b{b}',
            param_count=ae_params[b]['total_params'],
            config={'bottleneck_dim': b, 'epochs': epochs, 'batch_size': batch_size, 'lr': 1e-3, 'seed': seed},
            val_metric={'best_val_mse': ae_val_mses[b]},
            test_metric={'test_mse': test_mse},
            checkpoint_path=f'checkpoints/q3a_clean_ae_b{b}.pt',
            notes=f'Clean autoencoder with bottleneck capacity {b}.'
        )
        
    # Visual Reconstruction Grid for Q3(a)
    test_batch_imgs, _ = next(iter(unit_test))
    test_batch_np = test_batch_imgs.numpy()[:6]
    recons_dict = {}
    for b in bottlenecks:
        ae_models[b].eval()
        with torch.no_grad():
            rec = ae_models[b](test_batch_imgs[:6].to(device)).cpu().numpy()
            recons_dict[f'Bottleneck {b}'] = rec
            
    plot_q3a_reconstruction_grid(test_batch_np, recons_dict, num_samples=6, save_path='figures/q3a_reconstruction_grid.png')
    
    # -----------------------------------------------------------------
    # Q3(b): Denoising Autoencoder & Classification Relevance
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q3(b)] TRAINING DENOISING AUTOENCODER (Bottleneck = 256, Noise sigma = 0.2)...", flush=True)
    print("=" * 70, flush=True)
    
    dae = ConvAutoencoder(bottleneck_dim=256)
    dae_p_info = get_ae_parameter_counts(dae)
    dae_ckpt = 'checkpoints/q3b_denoising_ae.pt'
    dae_log = 'experiments/q3b_denoising_ae_log.json'
    
    if os.path.exists(dae_ckpt) and os.path.exists(dae_log):
        print(f"Loading existing DAE checkpoint: {dae_ckpt}", flush=True)
        dae.load_state_dict(torch.load(dae_ckpt, map_location=device))
        with open(dae_log, 'r') as f:
            dae_hist = json.load(f)
        dae_val_mse = dae_hist['best_val_mse']
    else:
        dae_hist = train_autoencoder(
            dae, unit_train, unit_val,
            epochs=epochs, lr=1e-3, device=device,
            checkpoint_path=dae_ckpt, log_path=dae_log, seed=seed,
            is_denoising=True, noise_sigma=noise_sigma
        )
        dae.load_state_dict(torch.load(dae_ckpt, map_location=device))
        dae_val_mse = dae_hist['best_val_mse']
        
    dae_test_mse = evaluate_autoencoder(dae, unit_test, device=device, is_denoising=True, noise_sigma=noise_sigma)
    print(f">> Denoising AE: Best Val MSE = {dae_val_mse:.6f} | Test MSE = {dae_test_mse:.6f}", flush=True)
    
    log_experiment(
        exp_id='Q3b_DAE_b256',
        question='Q3(b)',
        model_name='ConvAutoencoder_DAE_b256',
        param_count=dae_p_info['total_params'],
        config={'bottleneck_dim': 256, 'noise_sigma': noise_sigma, 'epochs': epochs, 'batch_size': batch_size, 'lr': 1e-3, 'seed': seed},
        val_metric={'best_val_mse': dae_val_mse},
        test_metric={'test_mse': dae_test_mse},
        checkpoint_path=dae_ckpt,
        notes='Denoising autoencoder trained with Gaussian noise sigma=0.2.'
    )
    
    # Downstream Classification via FIXED CNN CLASSIFIER from Q2(a)
    print("\nEvaluating Downstream Classification using FIXED CNN Classifier from Q2(a)...", flush=True)
    fixed_cnn = FairCNN(num_classes=10)
    fixed_cnn.load_state_dict(torch.load('checkpoints/q2a_fair_cnn_clean.pt', map_location=device))
    fixed_cnn.to(device)
    fixed_cnn.eval()
    
    # Normalization parameters matching classifier expectations
    cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    cifar_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    
    clean_ae = ae_models[best_b]
    clean_ae.eval()
    dae.eval()
    
    correct_clean = 0
    correct_corrupt = 0
    correct_clean_ae = 0
    correct_dae = 0
    total_samples = 0
    
    rng = np.random.RandomState(seed)
    
    # For visualization of denoising
    vis_clean = []
    vis_noisy = []
    vis_cae = []
    vis_dae = []
    
    with torch.no_grad():
        for i_batch, (images, targets) in enumerate(unit_test): # images in [0, 1]
            images, targets = images.to(device), targets.to(device)
            B = images.size(0)
            total_samples += B
            
            # Deterministic noise for test evaluation
            noise = torch.from_numpy(rng.normal(0, noise_sigma, images.shape).astype(np.float32)).to(device)
            corrupted = torch.clamp(images + noise, 0.0, 1.0)
            
            # Reconstructions
            recon_clean_ae = clean_ae(corrupted)
            recon_dae = dae(corrupted)
            
            if i_batch == 0:
                vis_clean = images[:6].cpu().numpy()
                vis_noisy = corrupted[:6].cpu().numpy()
                vis_cae = recon_clean_ae[:6].cpu().numpy()
                vis_dae = recon_dae[:6].cpu().numpy()
                
            # 1. Clean Images Classification
            std_clean = (images - cifar_mean) / cifar_std
            correct_clean += (fixed_cnn(std_clean).argmax(dim=1) == targets).sum().item()
            
            # 2. Corrupted Images Classification
            std_corrupt = (corrupted - cifar_mean) / cifar_std
            correct_corrupt += (fixed_cnn(std_corrupt).argmax(dim=1) == targets).sum().item()
            
            # 3. Clean-AE Reconstruction Classification
            std_recon_clean = (recon_clean_ae - cifar_mean) / cifar_std
            correct_clean_ae += (fixed_cnn(std_recon_clean).argmax(dim=1) == targets).sum().item()
            
            # 4. DAE Reconstruction Classification
            std_recon_dae = (recon_dae - cifar_mean) / cifar_std
            correct_dae += (fixed_cnn(std_recon_dae).argmax(dim=1) == targets).sum().item()
            
    acc_clean = correct_clean / total_samples
    acc_corrupt = correct_corrupt / total_samples
    acc_clean_ae = correct_clean_ae / total_samples
    acc_dae = correct_dae / total_samples
    
    print("\n" + "-" * 60, flush=True)
    print("Q3(b) DOWNSTREAM CLASSIFICATION RECOVERY RESULTS:", flush=True)
    print(f"1. Clean Images Baseline:           {acc_clean*100:.2f}%", flush=True)
    print(f"2. Corrupted Images Directly:       {acc_corrupt*100:.2f}% (Drop: {(acc_clean - acc_corrupt)*100:+.2f}%)", flush=True)
    print(f"3. Clean AE Reconstructions:        {acc_clean_ae*100:.2f}% (Recovery vs Corrupt: {(acc_clean_ae - acc_corrupt)*100:+.2f}%)", flush=True)
    print(f"4. Denoising AE Reconstructions:    {acc_dae*100:.2f}% (Recovery vs Corrupt: {(acc_dae - acc_corrupt)*100:+.2f}%)", flush=True)
    print("-" * 60, flush=True)
    
    # Save visual comparison for DAE
    plot_q3b_denoising_comparison(vis_clean, vis_noisy, vis_cae, vis_dae, num_samples=6, save_path='figures/q3b_denoising_comparison.png')
    
    # -----------------------------------------------------------------
    # Q3(c): Representation Quality vs. Class Information (No Fine-Tuning)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q3(c)] LINEAR PROBING ON FROZEN REPRESENTATIONS (NO ENCODER FINE-TUNING)", flush=True)
    print("=" * 70, flush=True)
    
    tradeoff_results = []
    probing_models = [
        ('b=64', ae_models[64], ae_test_mses[64]),
        ('b=256', ae_models[256], ae_test_mses[256]),
        ('b=1024', ae_models[1024], ae_test_mses[1024]),
        ('DAE_b256', dae, dae_test_mse)
    ]
    
    for name, model, test_mse in probing_models:
        print(f"Extracting frozen representations for {name}...", flush=True)
        train_x, train_y = extract_features(model, unit_train, device=device)
        val_x, val_y = extract_features(model, unit_val, device=device)
        test_x, test_y = extract_features(model, unit_test, device=device)
        
        val_acc, test_acc = train_linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, seed=seed)
        print(f">> {name}: Recon Test MSE: {test_mse:.6f} | Linear Probe Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%", flush=True)
        tradeoff_results.append({
            'name': name,
            'test_mse': test_mse,
            'val_acc': val_acc,
            'test_acc': test_acc
        })
        
    plot_q3c_tradeoff(
        bottlenecks=[64, 256, 1024],
        recon_mses=[r['test_mse'] for r in tradeoff_results[:3]],
        class_accuracies=[r['test_acc'] for r in tradeoff_results[:3]],
        save_path='figures/q3c_reconstruction_vs_class_info.pdf'
    )
    
    os.makedirs('results', exist_ok=True)
    with open('results/q3_summary.json', 'w') as f:
        json.dump({
            'bottleneck_mses': {str(b): {'val_mse': ae_val_mses[b], 'test_mse': ae_test_mses[b]} for b in bottlenecks},
            'selected_bottleneck': best_b,
            'downstream_classification': {
                'clean': acc_clean,
                'corrupted': acc_corrupt,
                'clean_ae_recon': acc_clean_ae,
                'dae_recon': acc_dae
            },
            'representation_tradeoff': tradeoff_results
        }, f, indent=2)
        
    print("\nQ3 EXPERIMENTS FULLY COMPLETE!", flush=True)

if __name__ == '__main__':
    main()
