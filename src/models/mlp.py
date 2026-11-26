import torch
import torch.nn as nn

class StandardMLP(nn.Module):
    """
    Standard fully connected MLP matching parameter budget ~112k.
    Architecture: 3072 -> 36 -> 36 -> 10.
    Total parameters: 112,330.
    """
    def __init__(self, input_dim=3072, hidden_dim=36, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class PatchLocallyConnectedMLP(nn.Module):
    """
    Locally Connected Network (LCN) operating on 16 non-overlapping 8x8 patches.
    Has strictly local receptive fields, but NO weight sharing across patches.
    Matches parameter budget: ~113.6k parameters.
    Used in Q1(a) to isolate local spatial connectivity from parameter efficiency/weight sharing.
    """
    def __init__(self, patch_size=8, hidden_per_patch=35, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2  # 16 patches
        patch_dim = 3 * patch_size * patch_size     # 192 pixels per patch
        
        # 16 independent linear layers (no weight sharing!)
        self.patch_linears = nn.ModuleList([
            nn.Linear(patch_dim, hidden_per_patch) for _ in range(self.num_patches)
        ])
        self.relu = nn.ReLU()
        self.head = nn.Linear(self.num_patches * hidden_per_patch, num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        # Extract 16 patches: (B, 16, 192)
        p = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        p = p.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, self.num_patches, 3 * self.patch_size * self.patch_size)
        
        patch_features = [self.relu(self.patch_linears[i](p[:, i, :])) for i in range(self.num_patches)]
        concat_features = torch.cat(patch_features, dim=1) # (B, 16 * 35 = 560)
        return self.head(concat_features)

class HighCapacityMLP(nn.Module):
    """
    Overparameterized MLP for Q1(b) memorization experiments.
    Architecture: 3072 -> 512 -> 512 -> 10.
    Total parameters: 1,841,162 (~1.84M parameters).
    Has sufficient capacity to memorize 45,000 CIFAR-10 images.
    """
    def __init__(self, input_dim=3072, hidden_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
