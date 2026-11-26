import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_svm_grid_sweep(train_feats, train_labels, val_feats, val_labels,
                       c_values=[0.01, 0.1, 1.0, 10.0, 100.0],
                       kernels=['linear', 'rbf', 'poly'],
                       seed=26013):
    """
    Performs the full C and kernel sweep.
    StandardScaler is fitted strictly on train_feats and applied to val_feats.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_feats)
    x_val = scaler.transform(val_feats)
    
    n_train = len(train_labels)
    sweep_results = []
    trained_models = {}
    
    print(f"Starting SVM Grid Sweep: {len(kernels)} kernels x {len(c_values)} C values = {len(kernels)*len(c_values)} configurations.")
    print(f"Training subset size: {n_train} samples, Val set size: {len(val_labels)} samples.")
    
    for k in kernels:
        for c in c_values:
            print(f"Training SVM [Kernel={k.upper():<6}, C={c:<6}] ... ", end="", flush=True)
            if k == 'poly':
                clf = SVC(C=c, kernel='poly', degree=3, random_state=seed)
            elif k == 'rbf':
                clf = SVC(C=c, kernel='rbf', gamma='scale', random_state=seed)
            else:
                clf = SVC(C=c, kernel='linear', random_state=seed)
                
            clf.fit(x_train, train_labels)
            
            train_acc = accuracy_score(train_labels, clf.predict(x_train))
            val_acc = accuracy_score(val_labels, clf.predict(x_val))
            n_sv = len(clf.support_)
            sv_fraction = n_sv / n_train
            
            print(f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | SV Fraction: {sv_fraction*100:5.2f}% ({n_sv}/{n_train} SVs)", flush=True)
            
            record = {
                'kernel': k,
                'C': float(c),
                'train_acc': float(train_acc),
                'val_acc': float(val_acc),
                'n_support_vectors': int(n_sv),
                'sv_fraction': float(sv_fraction),
                'train_size': int(n_train)
            }
            sweep_results.append(record)
            trained_models[(k, c)] = clf
            
    # Model selection based strictly on validation accuracy
    best_record = max(sweep_results, key=lambda x: x['val_acc'])
    best_model = trained_models[(best_record['kernel'], best_record['C'])]
    
    return sweep_results, best_record, best_model, scaler

def analyze_boundary_samples_for_pair(clf, scaler, train_feats, train_labels, train_indices,
                                      test_feats, test_labels, class_i, class_j,
                                      top_k=4):
    """
    For a given difficult class pair (class_i, class_j):
    1. Extracts training support vectors specifically for class_i and class_j.
    2. Identifies test samples closest to the decision boundary |d(x)| ~ 0.
    """
    x_test = scaler.transform(test_feats)
    decision_scores = clf.decision_function(x_test)
    classes = clf.classes_
    
    # Handle both OVR (N, 10) and OVO (N, 45) decision function formats
    if decision_scores.ndim == 2 and decision_scores.shape[1] == len(classes):
        # OVR format: pairwise relative distance between class_i and class_j
        pair_scores = decision_scores[:, class_i] - decision_scores[:, class_j]
    else:
        # One-vs-One format: find binary classifier index
        pair_idx = None
        curr_idx = 0
        for a in range(len(classes)):
            for b in range(a + 1, len(classes)):
                if (classes[a] == class_i and classes[b] == class_j) or (classes[a] == class_j and classes[b] == class_i):
                    pair_idx = curr_idx
                    break
                curr_idx += 1
            if pair_idx is not None:
                break
        pair_scores = decision_scores[:, pair_idx] if pair_idx is not None else (decision_scores[:, 0])
    
    # Filter test samples belonging to class_i or class_j
    mask = (test_labels == class_i) | (test_labels == class_j)
    test_indices_in_pair = np.where(mask)[0]
    pair_test_scores = pair_scores[test_indices_in_pair]
    
    # Sort by absolute distance to boundary
    abs_dists = np.abs(pair_test_scores)
    sorted_order = np.argsort(abs_dists)
    closest_test_indices = test_indices_in_pair[sorted_order[:top_k]]
    closest_scores = pair_test_scores[sorted_order[:top_k]]
    
    # Extract training support vectors specifically for class_i and class_j
    sv_indices_in_subset = clf.support_ # indices in x_train
    sv_labels = train_labels[sv_indices_in_subset]
    
    sv_ci_indices = sv_indices_in_subset[sv_labels == class_i]
    sv_cj_indices = sv_indices_in_subset[sv_labels == class_j]
    
    # Pick 2 SVs from class_i and 2 SVs from class_j
    selected_sv_sub = np.concatenate([sv_ci_indices[:2], sv_cj_indices[:2]])
    sv_train_global_indices = train_indices[selected_sv_sub]
    
    return {
        'closest_test_indices': closest_test_indices.tolist(),
        'closest_decision_scores': closest_scores.tolist(),
        'sv_global_indices': sv_train_global_indices.tolist(),
        'num_sv_class_i': int(len(sv_ci_indices)),
        'num_sv_class_j': int(len(sv_cj_indices))
    }
