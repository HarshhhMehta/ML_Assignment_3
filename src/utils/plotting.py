import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Publication styling for CVPR (clean, legible font sizes, tight bounding box)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def plot_q1a_curves(standard_history, patch_history, save_path='figures/q1a_learning_curves.pdf'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    
    epochs = range(1, len(standard_history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, standard_history['train_loss'], 'b--', label='Standard MLP (Train)')
    axes[0].plot(epochs, standard_history['val_loss'], 'b-', label='Standard MLP (Val)')
    axes[0].plot(epochs, patch_history['train_loss'], 'r--', label='Patch-LCN (Train)')
    axes[0].plot(epochs, patch_history['val_loss'], 'r-', label='Patch-LCN (Val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss Trajectory')
    axes[0].legend()
    
    # Accuracy curves
    axes[1].plot(epochs, np.array(standard_history['train_acc']) * 100, 'b--', label='Standard MLP (Train)')
    axes[1].plot(epochs, np.array(standard_history['val_acc']) * 100, 'b-', label='Standard MLP (Val)')
    axes[1].plot(epochs, np.array(patch_history['train_acc']) * 100, 'r--', label='Patch-LCN (Train)')
    axes[1].plot(epochs, np.array(patch_history['val_acc']) * 100, 'r-', label='Patch-LCN (Val)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Trajectory')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q1(a) curves to {save_path}.")

def plot_q1b_memorization(history, save_path='figures/q1b_memorization_trajectory.pdf'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    epochs = history['epoch']
    train_acc = np.array(history['train_acc']) * 100
    val_acc = np.array(history['val_acc']) * 100
    gen_gap = np.array(history['generalization_gap']) * 100
    
    ax1.plot(epochs, train_acc, 'r-o', markersize=3, label='Train Acc (50% Corrupted)')
    ax1.plot(epochs, val_acc, 'g-s', markersize=3, label='Val Acc (100% Clean)')
    ax1.plot(epochs, gen_gap, 'm--', label='Generalization Gap')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy / Gap (%)')
    ax1.set_title('Q1(b): MLP Memorization Dynamics Under Label Noise')
    ax1.legend(loc='center right')
    
    # Highlight early learning phase vs memorization phase
    max_val_epoch = np.argmax(val_acc) + 1
    ax1.axvline(x=max_val_epoch, color='gray', linestyle=':', label=f'Peak Val (Ep {max_val_epoch})')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q1(b) memorization plot to {save_path}.")

def plot_q2_comparisons(results_dict, save_path='figures/q2_spatial_disturbance.pdf'):
    """
    results_dict: {
        'MLP': {'Clean': acc, 'Pixel Permuted': acc, 'Patch Permuted': acc},
        'CNN': {'Clean': acc, 'Pixel Permuted': acc, 'Patch Permuted': acc}
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    conditions = ['Clean', 'Patch Permuted', 'Pixel Permuted']
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(6, 4))
    mlp_accs = [results_dict['MLP'][c] * 100 for c in conditions]
    cnn_accs = [results_dict['CNN'][c] * 100 for c in conditions]
    
    rects1 = ax.bar(x - width/2, mlp_accs, width, label='MLP (~112k params)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, cnn_accs, width, label='CNN (~112k params)', color='#DD8452')
    
    ax.set_ylabel('Official Test Accuracy (%)')
    ax.set_title('Q2: Impact of Spatial Disruption on CNN vs. MLP')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q2 comparison bar plot to {save_path}.")

def plot_q2c_confidence_histograms(clean_res, disturbed_res, save_path='figures/q2c_confidence_analysis.pdf'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    
    # Clean condition
    conf_clean_corr = clean_res['confidences'][clean_res['predictions'] == clean_res['targets']]
    conf_clean_inc = clean_res['confidences'][clean_res['predictions'] != clean_res['targets']]
    
    axes[0].hist(conf_clean_corr, bins=20, alpha=0.7, color='green', label=f'Correct (Mean={np.mean(conf_clean_corr):.2f})', density=True)
    axes[0].hist(conf_clean_inc, bins=20, alpha=0.7, color='red', label=f'Incorrect (Mean={np.mean(conf_clean_inc):.2f})', density=True)
    axes[0].set_title(f"Clean CIFAR-10 (ECE: {clean_res['ece']:.3f})")
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Disturbed condition
    conf_dist_corr = disturbed_res['confidences'][disturbed_res['predictions'] == disturbed_res['targets']]
    conf_dist_inc = disturbed_res['confidences'][disturbed_res['predictions'] != disturbed_res['targets']]
    
    axes[1].hist(conf_dist_corr, bins=20, alpha=0.7, color='green', label=f'Correct (Mean={np.mean(conf_dist_corr):.2f})', density=True)
    axes[1].hist(conf_dist_inc, bins=20, alpha=0.7, color='red', label=f'Incorrect (Mean={np.mean(conf_dist_inc):.2f})', density=True)
    axes[1].set_title(f"Spatially Disturbed (ECE: {disturbed_res['ece']:.3f})")
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q2(c) confidence analysis plot to {save_path}.")

def plot_q3a_reconstruction_grid(originals, reconstructions_dict, num_samples=6, save_path='figures/q3a_reconstruction_grid.png'):
    """
    originals: numpy array (N, 3, 32, 32)
    reconstructions_dict: {'b=64': array, 'b=256': array, 'b=1024': array}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    keys = list(reconstructions_dict.keys())
    num_rows = 1 + len(keys)
    
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(num_samples * 1.5, num_rows * 1.5))
    
    for i in range(num_samples):
        # Original
        orig = np.transpose(originals[i], (1, 2, 0))
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', loc='left', fontsize=9)
            
        # Reconstructions
        for row_idx, k in enumerate(keys, start=1):
            recon = np.transpose(reconstructions_dict[k][i], (1, 2, 0))
            axes[row_idx, i].imshow(np.clip(recon, 0, 1))
            axes[row_idx, i].axis('off')
            if i == 0:
                axes[row_idx, i].set_title(k, loc='left', fontsize=9)
                
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q3(a) reconstruction visual grid to {save_path}.")

def plot_q3b_denoising_comparison(clean_imgs, noisy_imgs, clean_ae_recons, dae_recons, num_samples=6, save_path='figures/q3b_denoising_comparison.png'):
    """
    Compares 4 rows:
    Row 0: Ground Truth Clean Images
    Row 1: Corrupted Inputs (Gaussian noise sigma=0.2)
    Row 2: Clean AE Reconstructions (passes noise through)
    Row 3: Denoising AE Reconstructions (removes noise, restores semantic contours)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 1.5, 4 * 1.5))
    row_titles = ['Clean Ground Truth', 'Corrupted (σ=0.2)', 'Clean AE Recon', 'Denoising AE Recon']
    
    for i in range(num_samples):
        # Clean
        c_img = np.transpose(clean_imgs[i], (1, 2, 0))
        axes[0, i].imshow(np.clip(c_img, 0, 1))
        axes[0, i].axis('off')
        
        # Noisy
        n_img = np.transpose(noisy_imgs[i], (1, 2, 0))
        axes[1, i].imshow(np.clip(n_img, 0, 1))
        axes[1, i].axis('off')
        
        # Clean AE
        cae_img = np.transpose(clean_ae_recons[i], (1, 2, 0))
        axes[2, i].imshow(np.clip(cae_img, 0, 1))
        axes[2, i].axis('off')
        
        # DAE
        dae_img = np.transpose(dae_recons[i], (1, 2, 0))
        axes[3, i].imshow(np.clip(dae_img, 0, 1))
        axes[3, i].axis('off')
        
        if i == 0:
            for r, title in enumerate(row_titles):
                axes[r, i].set_title(title, loc='left', fontsize=9)
                
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q3(b) denoising comparison visual grid to {save_path}.")

def plot_q3c_tradeoff(bottlenecks, recon_mses, class_accuracies, save_path='figures/q3c_reconstruction_vs_class_info.pdf'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(bottlenecks))
    ax1.plot(x, recon_mses, 'b-o', label='Reconstruction MSE (Lower is Better)')
    ax1.set_xlabel('Bottleneck Dimension')
    ax1.set_ylabel('Test Reconstruction MSE', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in bottlenecks])
    
    ax2 = ax1.twinx()
    ax2.plot(x, [acc * 100 for acc in class_accuracies], 'r-s', label='Linear Probe Acc (%) (Higher is Better)')
    ax2.set_ylabel('Test Classification Accuracy (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Q3(c): Reconstruction Quality vs. Class Information')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q3(c) tradeoff plot to {save_path}.")

def plot_q4a_svm_sweep(sweep_results, save_path='figures/q4a_svm_sweep.pdf'):
    """
    sweep_results: list of dicts with keys: 'kernel', 'C', 'val_acc', 'sv_fraction'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    
    kernels = ['linear', 'rbf', 'poly']
    colors = {'linear': '#4C72B0', 'rbf': '#DD8452', 'poly': '#55A868'}
    markers = {'linear': 'o', 'rbf': 's', 'poly': '^'}
    
    for k in kernels:
        k_results = [r for r in sweep_results if r['kernel'] == k]
        k_results.sort(key=lambda x: x['C'])
        c_vals = [r['C'] for r in k_results]
        val_accs = [r['val_acc'] * 100 for r in k_results]
        sv_fracs = [r['sv_fraction'] * 100 for r in k_results]
        
        ax1.plot(c_vals, val_accs, marker=markers[k], color=colors[k], label=k.upper())
        ax2.plot(c_vals, sv_fracs, marker=markers[k], color=colors[k], label=k.upper())
        
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization Parameter C (log scale)')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy vs. C')
    ax1.legend()
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Parameter C (log scale)')
    ax2.set_ylabel('Support Vector Fraction (%)')
    ax2.set_title('Support Vector Sparsity vs. C')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q4(a) SVM sweep curves to {save_path}.")

def plot_q4b_hard_pairs(train_imgs, test_imgs, pairs_data, save_path='figures/q4b_hard_pairs.png'):
    """
    Visualizes the top 3 difficult class pairs:
    For each pair (row):
    - 2 Training Support Vectors for Class A
    - 2 Training Support Vectors for Class B
    - 2 Closest Test Samples near Decision Boundary (with decision scores d(x))
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_pairs = len(pairs_data)
    cols = 6
    fig, axes = plt.subplots(num_pairs, cols, figsize=(cols * 1.6, num_pairs * 2.0))
    if num_pairs == 1:
        axes = np.expand_dims(axes, 0)
        
    for p_idx, p in enumerate(pairs_data):
        info = p['pair_info']
        geom = p['boundary_geometry']
        name_i = info['name_i']
        name_j = info['name_j']
        
        # 1. SV images
        sv_indices = geom['sv_global_indices']
        for s_idx in range(min(4, len(sv_indices))):
            sv_img = np.transpose(train_imgs[sv_indices[s_idx]], (1, 2, 0)) / 255.0 if train_imgs[sv_indices[s_idx]].max() > 1.0 else np.transpose(train_imgs[sv_indices[s_idx]], (1, 2, 0))
            axes[p_idx, s_idx].imshow(np.clip(sv_img, 0, 1))
            axes[p_idx, s_idx].axis('off')
            lbl = f"SV ({name_i if s_idx < 2 else name_j})"
            axes[p_idx, s_idx].set_title(lbl, fontsize=8)
            
        # 2. Test boundary images
        test_indices = geom['closest_test_indices']
        scores = geom['closest_decision_scores']
        for t_idx in range(min(2, len(test_indices))):
            col_pos = 4 + t_idx
            t_img = np.transpose(test_imgs[test_indices[t_idx]], (1, 2, 0)) / 255.0 if test_imgs[test_indices[t_idx]].max() > 1.0 else np.transpose(test_imgs[test_indices[t_idx]], (1, 2, 0))
            axes[p_idx, col_pos].imshow(np.clip(t_img, 0, 1))
            axes[p_idx, col_pos].axis('off')
            sc = scores[t_idx]
            axes[p_idx, col_pos].set_title(f"Test |d|={abs(sc):.2f}", fontsize=8)
            
        # Row label
        axes[p_idx, 0].set_ylabel(f"{name_i.upper()} vs\n{name_j.upper()}\n({info['total_symmetric_error']} errs)", fontsize=9, rotation=0, labelpad=40, va='center')
        
    plt.suptitle("Q4(b): Support Vectors and Boundary Examples for Top Confused Class Pairs", fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q4(b) hard pairs boundary visualization to {save_path}.")

def plot_q4c_contingency_heatmap(contingency_matrix, class_names, save_path='figures/q4c_contingency_matrix.pdf'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    mat = np.array(contingency_matrix)
    cax = ax.matshow(mat, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels([f'Cluster {i}' for i in range(mat.shape[0])])
    
    ax.set_xlabel('True Ground-Truth Class')
    ax.set_ylabel('Unsupervised Cluster Assignment')
    ax.set_title('Q4(c): Test Set Cluster-to-Class Contingency Table', pad=20)
    
    # Annotate numbers
    for i in range(mat.shape[0]):
        for j in range(len(class_names)):
            val = mat[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        color='white' if val > mat.max()/2 else 'black', fontsize=8)
                        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Q4(c) contingency table heatmap to {save_path}.")
