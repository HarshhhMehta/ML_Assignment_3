import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

def compute_confidence_and_calibration(model, loader, device='cpu', n_bins=15):
    """
    Computes:
    - Overall test accuracy
    - Mean and Max confidence across all predictions
    - Confidence of CORRECT predictions vs INCORRECT predictions
    - Expected Calibration Error (ECE)
    - Full prediction and probability arrays
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0) # (N, 10)
    all_preds = np.concatenate(all_preds, axis=0) # (N,)
    all_targets = np.concatenate(all_targets, axis=0) # (N,)
    
    confidences = np.max(all_probs, axis=1) # (N,)
    correct_mask = (all_preds == all_targets)
    
    acc = np.mean(correct_mask)
    conf_correct = np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0.0
    conf_incorrect = np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0.0
    overall_conf = np.mean(confidences)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            acc_in_bin = np.mean(correct_mask[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_stats.append({
                'bin_range': (float(bin_lower), float(bin_upper)),
                'accuracy': float(acc_in_bin),
                'confidence': float(conf_in_bin),
                'count': int(np.sum(in_bin))
            })
            
    return {
        'accuracy': float(acc),
        'overall_confidence': float(overall_conf),
        'confidence_correct': float(conf_correct),
        'confidence_incorrect': float(conf_incorrect),
        'ece': float(ece),
        'bin_stats': bin_stats,
        'predictions': all_preds,
        'targets': all_targets,
        'confidences': confidences
    }

def compute_clustering_metrics(true_labels, cluster_assignments, num_classes=10):
    """
    Computes:
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Cluster-to-Class Contingency Table (num_clusters x num_classes)
    """
    ari = adjusted_rand_score(true_labels, cluster_assignments)
    nmi = normalized_mutual_info_score(true_labels, cluster_assignments)
    
    # Contingency matrix: rows = clusters, cols = true classes
    unique_clusters = np.unique(cluster_assignments)
    num_clusters = len(unique_clusters)
    contingency = np.zeros((num_clusters, num_classes), dtype=np.int64)
    
    for i, c in enumerate(unique_clusters):
        mask = (cluster_assignments == c)
        for cls_idx in range(num_classes):
            contingency[i, cls_idx] = np.sum(true_labels[mask] == cls_idx)
            
    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'contingency_table': contingency.tolist(),
        'cluster_purities': [float(np.max(row) / np.sum(row)) if np.sum(row) > 0 else 0.0 for row in contingency]
    }

def identify_difficult_class_pairs(true_labels, predicted_labels, class_names=None, top_k=3):
    """
    Identifies class pairs with highest symmetric confusion: E_ij = N(i->j) + N(j->i).
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    num_classes = cm.shape[0]
    pair_errors = []
    
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            err = cm[i, j] + cm[j, i]
            pair_errors.append((err, i, j, cm[i, j], cm[j, i]))
            
    pair_errors.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pair_errors[:top_k]
    
    results = []
    for err, i, j, err_ij, err_ji in top_pairs:
        name_i = class_names[i] if class_names else str(i)
        name_j = class_names[j] if class_names else str(j)
        results.append({
            'class_i': int(i),
            'class_j': int(j),
            'name_i': name_i,
            'name_j': name_j,
            'total_symmetric_error': int(err),
            'error_i_as_j': int(err_ij),
            'error_j_as_i': int(err_ji)
        })
    return results
