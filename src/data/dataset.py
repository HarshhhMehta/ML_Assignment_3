import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Official CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        dict_entry = pickle.load(f, encoding='bytes')
    data = dict_entry[b'data']  # shape: (10000, 3072)
    labels = dict_entry[b'labels']  # list of 10000 ints
    # reshape to (10000, 3, 32, 32)
    data = data.reshape(10000, 3, 32, 32)
    return data, np.array(labels, dtype=np.int64)

def load_raw_cifar10(root_dir='data/raw/cifar-10-batches-py'):
    train_data_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch_path = os.path.join(root_dir, f'data_batch_{i}')
        data, labels = load_cifar_batch(batch_path)
        train_data_list.append(data)
        train_labels_list.append(labels)
    
    train_data = np.concatenate(train_data_list, axis=0) # (50000, 3, 32, 32)
    train_labels = np.concatenate(train_labels_list, axis=0) # (50000,)
    
    test_batch_path = os.path.join(root_dir, 'test_batch')
    test_data, test_labels = load_cifar_batch(test_batch_path) # (10000, 3, 32, 32)
    
    return train_data, train_labels, test_data, test_labels

def get_or_create_stratified_split(train_labels, split_dir='data/split_indices', seed=26013, val_per_class=500):
    os.makedirs(split_dir, exist_ok=True)
    train_idx_path = os.path.join(split_dir, 'train_indices.npy')
    val_idx_path = os.path.join(split_dir, 'val_indices.npy')
    
    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path):
        train_indices = np.load(train_idx_path)
        val_indices = np.load(val_idx_path)
        return train_indices, val_indices
    
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    
    num_classes = len(np.unique(train_labels))
    for c in range(num_classes):
        class_indices = np.where(train_labels == c)[0]
        rng.shuffle(class_indices)
        val_part = class_indices[:val_per_class]
        train_part = class_indices[val_per_class:]
        val_indices.extend(val_part)
        train_indices.extend(train_part)
        
    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    
    # Shuffle the aggregated lists deterministically
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    np.save(train_idx_path, train_indices)
    np.save(val_idx_path, val_indices)
    print(f"Generated stratified split with seed {seed}: {len(train_indices)} train, {len(val_indices)} val.")
    return train_indices, val_indices

def get_or_create_svm_subset(train_labels, train_indices, subset_dir='data/subset_indices', seed=26013, samples_per_class=500):
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = os.path.join(subset_dir, 'q4_svm_subset_indices.npy')
    if os.path.exists(subset_path):
        return np.load(subset_path)
    
    rng = np.random.RandomState(seed)
    subset_indices = []
    num_classes = len(np.unique(train_labels))
    
    # Subset must be drawn strictly from the 45,000 train_indices
    train_labels_subset = train_labels[train_indices]
    for c in range(num_classes):
        class_pos = np.where(train_labels_subset == c)[0]
        rng.shuffle(class_pos)
        selected_pos = class_pos[:samples_per_class]
        subset_indices.extend(train_indices[selected_pos])
        
    subset_indices = np.array(subset_indices, dtype=np.int64)
    rng.shuffle(subset_indices)
    np.save(subset_path, subset_indices)
    print(f"Generated Q4 SVM stratified subset with seed {seed}: {len(subset_indices)} samples ({samples_per_class}/class).")
    return subset_indices

class CIFAR10Dataset(Dataset):
    """
    Modular CIFAR10 Dataset supporting:
    - Normalization: 'standard' (zero mean, unit std) or 'unit' ([0, 1] for autoencoders)
    - Transform: custom functional transform (e.g. pixel permutation, patch permutation, noise)
    - Corrupt targets: for Q1(b) memorization, replacing labels with random ones
    """
    def __init__(self, data, labels, normalize='standard', transform=None, corrupted_labels=None):
        # data: shape (N, 3, 32, 32), uint8
        self.data = data
        self.labels = corrupted_labels if corrupted_labels is not None else labels
        self.raw_labels = labels
        self.normalize = normalize
        self.transform = transform
        
        # Precompute mean and std tensors
        if self.normalize == 'standard':
            self.mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
            self.std = np.array(CIFAR10_STD, dtype=np.float32).reshape(3, 1, 1)
        elif self.normalize == 'unit':
            self.mean = 0.0
            self.std = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert uint8 [0, 255] to float32 [0.0, 1.0]
        img = self.data[idx].astype(np.float32) / 255.0
        
        # Apply spatial/noise transform if provided
        if self.transform is not None:
            img = self.transform(img)
            
        # Normalization
        if self.normalize == 'standard':
            img = (img - self.mean) / self.std
            
        tensor_img = torch.from_numpy(img).float()
        target = self.labels[idx]
        return tensor_img, target

def get_dataloaders(root_dir='data/raw/cifar-10-batches-py',
                    batch_size=64,
                    normalize='standard',
                    transform=None,
                    seed=26013,
                    num_workers=0):
    train_data_raw, train_labels_raw, test_data_raw, test_labels_raw = load_raw_cifar10(root_dir)
    train_idx, val_idx = get_or_create_stratified_split(train_labels_raw, seed=seed)
    
    train_set = CIFAR10Dataset(train_data_raw[train_idx], train_labels_raw[train_idx], normalize=normalize, transform=transform)
    val_set = CIFAR10Dataset(train_data_raw[val_idx], train_labels_raw[val_idx], normalize=normalize, transform=transform)
    test_set = CIFAR10Dataset(test_data_raw, test_labels_raw, normalize=normalize, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
