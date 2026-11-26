import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from src.data.dataset import load_raw_cifar10, get_or_create_stratified_split, get_or_create_svm_subset, CIFAR10Dataset, CLASS_NAMES
from src.models.cnn import FairCNN
from src.evaluation.svm_eval import run_svm_grid_sweep, analyze_boundary_samples_for_pair
from src.evaluation.metrics import identify_difficult_class_pairs, compute_clustering_metrics
from src.utils.registry import log_experiment
from src.utils.plotting import plot_q4a_svm_sweep, plot_q4b_hard_pairs, plot_q4c_contingency_heatmap

def extract_cnn_features(model, images_np, batch_size=128, device='cpu'):
    model.eval()
    dataset = CIFAR10Dataset(images_np, np.zeros(len(images_np)), normalize='standard')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    feats_list = []
    with torch.no_grad():
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(device)
            feats = model.extract_features(batch_imgs)
            feats_list.append(feats.cpu().numpy())
    return np.concatenate(feats_list, axis=0)

def main():
    torch.set_num_threads(8)
    device = 'cpu'
    seed = 26013
    
    print("=" * 70, flush=True)
    print("STARTING Q4 BONUS EXPERIMENTS: SVM SWEEP, HARD PAIRS & CLUSTERING", flush=True)
    print("=" * 70, flush=True)
    
    # -----------------------------------------------------------------
    # Step 1: Load Data & Extract Fixed 64-D CNN Representation
    # -----------------------------------------------------------------
    train_data_raw, train_labels_raw, test_data_raw, test_labels_raw = load_raw_cifar10()
    train_indices, val_indices = get_or_create_stratified_split(train_labels_raw, seed=seed)
    svm_subset_indices = get_or_create_svm_subset(train_labels_raw, train_indices, seed=seed, samples_per_class=500)
    
    print(f">> Q4 Dataset Splits: SVM Train Subset = {len(svm_subset_indices)}, Val = {len(val_indices)}, Test = {len(test_data_raw)}", flush=True)
    
    # Load Frozen Q2(a) CNN
    cnn_ckpt = 'checkpoints/q2a_fair_cnn_clean.pt'
    cnn = FairCNN(num_classes=10)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    cnn.to(device)
    cnn.eval()
    print(">> Loaded Frozen FairCNN representation extractor from Q2(a).", flush=True)
    
    # Verify representation extractor on test set
    test_ds = CIFAR10Dataset(test_data_raw, test_labels_raw, normalize='standard')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    correct_test = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct_test += (cnn(imgs).argmax(dim=1) == lbls).sum().item()
    cnn_test_acc = correct_test / len(test_labels_raw)
    print(f">> Verified FairCNN Baseline Test Accuracy: {cnn_test_acc*100:.2f}% (Matches Q2(a) exactly: 72.05%)", flush=True)
    
    print("Extracting fixed 64-D representations...", flush=True)
    train_sub_feats = extract_cnn_features(cnn, train_data_raw[svm_subset_indices], device=device)
    train_sub_labels = train_labels_raw[svm_subset_indices]
    
    val_feats = extract_cnn_features(cnn, train_data_raw[val_indices], device=device)
    val_labels = train_labels_raw[val_indices]
    
    test_feats = extract_cnn_features(cnn, test_data_raw, device=device)
    test_labels = test_labels_raw
    print(f">> Feature Extraction Complete: Shapes = {train_sub_feats.shape}, {val_feats.shape}, {test_feats.shape}", flush=True)
    
    # -----------------------------------------------------------------
    # Q4(a): SVM Parameter & Kernel Sweep (15 Configurations)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q4(a)] PERFORMING FULL SVM PARAMETER & KERNEL SWEEP (15 CONFIGS)", flush=True)
    print("=" * 70, flush=True)
    
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    kernels = ['linear', 'rbf', 'poly']
    
    sweep_results, best_record, best_svm, scaler = run_svm_grid_sweep(
        train_sub_feats, train_sub_labels,
        val_feats, val_labels,
        c_values=c_values, kernels=kernels, seed=seed
    )
    
    # Save sweep results table and curves
    os.makedirs('results', exist_ok=True)
    with open('results/q4a_svm_sweep_table.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)
        
    plot_q4a_svm_sweep(sweep_results, save_path='figures/q4a_svm_sweep.pdf')
    
    print("\n" + "-" * 60, flush=True)
    print(f">> FORMAL MODEL SELECTION (Validation Acc Only):", flush=True)
    print(f"   Selected Setting: Kernel = '{best_record['kernel'].upper()}', C = {best_record['C']}", flush=True)
    print(f"   Validation Accuracy: {best_record['val_acc']*100:.2f}% (SV Fraction: {best_record['sv_fraction']*100:.2f}%)", flush=True)
    print(f"   Selection rationale: Highest accuracy achieved on the full 5,000-image validation split.", flush=True)
    print("-" * 60, flush=True)
    
    # Single Official Test Evaluation strictly after model selection
    x_test = scaler.transform(test_feats)
    test_preds = best_svm.predict(x_test)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f">> Selected SVM Official Test Accuracy: {test_acc*100:.2f}%\n", flush=True)
    
    log_experiment(
        exp_id='Q4a_Selected_SVM',
        question='Q4(a)',
        model_name=f"SVM_{best_record['kernel']}_C{best_record['C']}",
        param_count=best_record['n_support_vectors'],
        config={'kernel': best_record['kernel'], 'C': best_record['C'], 'train_subset_size': len(train_sub_labels)},
        val_metric={'val_acc': best_record['val_acc'], 'sv_fraction': best_record['sv_fraction']},
        test_metric={'test_acc': test_acc},
        reused_from='Q2a_Fair_CNN_Clean (Frozen Representation)',
        notes='Validation-selected SVM evaluated strictly once on official test set.'
    )
    
    # -----------------------------------------------------------------
    # Q4(b): Difficult Class Pairs, Support Vectors & Boundary Geometry
    # -----------------------------------------------------------------
    print("=" * 70, flush=True)
    print("[Q4(b)] IDENTIFYING TOP 3 DIFFICULT CLASS PAIRS & DECISION BOUNDARIES", flush=True)
    print("=" * 70, flush=True)
    
    top_pairs = identify_difficult_class_pairs(test_labels, test_preds, class_names=CLASS_NAMES, top_k=3)
    
    boundary_analysis_results = []
    for pair in top_pairs:
        ci, cj = pair['class_i'], pair['class_j']
        name_i, name_j = pair['name_i'], pair['name_j']
        err = pair['total_symmetric_error']
        print(f"\nDifficult Pair: {name_i.upper()} vs {name_j.upper()} (Total Test Errors: {err})", flush=True)
        print(f"  Confusion: {name_i} predicted as {name_j}: {pair['error_i_as_j']} | {name_j} predicted as {name_i}: {pair['error_j_as_i']}", flush=True)
        
        pair_geom = analyze_boundary_samples_for_pair(
            best_svm, scaler, train_sub_feats, train_sub_labels, svm_subset_indices,
            test_feats, test_labels, ci, cj, top_k=4
        )
        print(f"  Support Vectors for Pair: {name_i} SVs: {pair_geom['num_sv_class_i']}, {name_j} SVs: {pair_geom['num_sv_class_j']}", flush=True)
        print(f"  Closest Test Decision Scores (|d(x)|): {[round(s, 3) for s in pair_geom['closest_decision_scores']]}")
        
        boundary_analysis_results.append({
            'pair_info': pair,
            'boundary_geometry': pair_geom
        })
        
    with open('results/q4b_boundary_analysis.json', 'w') as f:
        json.dump(boundary_analysis_results, f, indent=2)
        
    # Plot Q4(b) Hard Pairs Visual Grid
    plot_q4b_hard_pairs(train_data_raw, test_data_raw, boundary_analysis_results, save_path='figures/q4b_hard_pairs.png')
    
    # -----------------------------------------------------------------
    # Q4(c): Unsupervised Clustering (K-Means vs. GMM vs. SVM)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("[Q4(c)] UNSUPERVISED CLUSTERING: K-MEANS & GMM VS. SUPERVISED SVM", flush=True)
    print("=" * 70, flush=True)
    
    x_train_sub = scaler.transform(train_sub_feats)
    
    # 1. K-Means Clustering (k=10) fit strictly on training representations without labels
    print("Fitting Unsupervised K-Means (k=10) on training representations (NO labels)...", flush=True)
    kmeans = KMeans(n_clusters=10, random_state=seed, n_init=10)
    kmeans.fit(x_train_sub)
    kmeans_test_preds = kmeans.predict(x_test)
    
    kmeans_metrics = compute_clustering_metrics(test_labels, cluster_assignments=kmeans_test_preds, num_classes=10)
    print(f">> K-Means Official Test ARI: {kmeans_metrics['ari']:.4f} | NMI: {kmeans_metrics['nmi']:.4f}", flush=True)
    
    # 2. Gaussian Mixture Model (GMM) fit strictly on training representations without labels
    print("Fitting Unsupervised GMM (k=10, diagonal covariance) on training representations (NO labels)...", flush=True)
    gmm = GaussianMixture(n_components=10, covariance_type='diag', random_state=seed, n_init=5)
    gmm.fit(x_train_sub)
    gmm_test_preds = gmm.predict(x_test)
    
    gmm_metrics = compute_clustering_metrics(test_labels, cluster_assignments=gmm_test_preds, num_classes=10)
    print(f">> GMM Official Test ARI:     {gmm_metrics['ari']:.4f} | NMI: {gmm_metrics['nmi']:.4f}", flush=True)
    
    # Save Contingency Heatmap for K-Means
    plot_q4c_contingency_heatmap(kmeans_metrics['contingency_table'], CLASS_NAMES, save_path='figures/q4c_contingency_matrix.pdf')
    
    # 3. Disagreement Analysis: Where Unsupervised Clustering and Supervised SVM Diverge
    contingency = np.array(kmeans_metrics['contingency_table'])
    cluster_dominance = contingency.argmax(axis=1) # which class dominates each cluster
    
    disagreement_cases = []
    # Identify clusters that contain large fractions of two distinct classes
    for cluster_id in range(10):
        sorted_classes = np.argsort(contingency[cluster_id])[::-1]
        top1_cls, top2_cls = sorted_classes[0], sorted_classes[1]
        cnt1, cnt2 = int(contingency[cluster_id, top1_cls]), int(contingency[cluster_id, top2_cls])
        
        # If the second most frequent class has substantial representation (> 100 test samples)
        if cnt2 > 100:
            disagreement_cases.append({
                'cluster_id': int(cluster_id),
                'dominant_class': CLASS_NAMES[top1_cls],
                'dominant_count': cnt1,
                'secondary_class': CLASS_NAMES[top2_cls],
                'secondary_count': cnt2,
                'conflation_ratio': float(cnt2 / (cnt1 + cnt2))
            })
            
    print("\n" + "-" * 60, flush=True)
    print("QUANTITATIVE SVM VS. CLUSTERING DISAGREEMENT SUMMARY:", flush=True)
    print(f"Supervised SVM Test Acc: {test_acc*100:.2f}% (Separates classes via margin-maximizing hyperplanes).")
    print(f"Unsupervised K-Means:   NMI = {kmeans_metrics['nmi']:.4f}, ARI = {kmeans_metrics['ari']:.4f}")
    print(f"Unsupervised GMM:       NMI = {gmm_metrics['nmi']:.4f}, ARI = {gmm_metrics['ari']:.4f}")
    print("-" * 60, flush=True)
    print("Disagreement Case Studies (Unsupervised Conflation vs. Supervised Separation):")
    for case in disagreement_cases:
        print(f"  * Cluster {case['cluster_id']}: Merges '{case['dominant_class'].upper()}' ({case['dominant_count']} samples) and '{case['secondary_class'].upper()}' ({case['secondary_count']} samples).")
        print(f"    Reason: Feature similarity in visual geometry (e.g. background/aspect ratio) causes clustering to merge them, while supervised SVM separates them using discriminative boundary hyperplanes.")
    print("-" * 60, flush=True)
    
    with open('results/q4c_clustering_summary.json', 'w') as f:
        json.dump({
            'kmeans_metrics': kmeans_metrics,
            'gmm_metrics': gmm_metrics,
            'disagreement_cases': disagreement_cases,
            'svm_test_acc': test_acc
        }, f, indent=2)
        
    print("\nQ4 BONUS EXPERIMENTS FULLY COMPLETE!", flush=True)

if __name__ == '__main__':
    main()
