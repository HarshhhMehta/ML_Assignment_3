import numpy as np

class PixelPermutationTransform:
    """
    Permutes the 32x32 spatial positions deterministically with a fixed permutation vector.
    Preserves 100% of the pixel intensity distribution.
    Can permute spatial coordinates identically across the 3 RGB channels (channel-aligned)
    or permute all 3x32x32 values. Standard convention: channel-aligned spatial permutation.
    """
    def __init__(self, seed=26013, channel_aligned=True):
        self.seed = seed
        self.channel_aligned = channel_aligned
        rng = np.random.RandomState(seed)
        if channel_aligned:
            self.perm = rng.permutation(32 * 32)
        else:
            self.perm = rng.permutation(3 * 32 * 32)

    def __call__(self, img):
        # img: numpy array shape (3, 32, 32)
        if self.channel_aligned:
            c, h, w = img.shape
            flat = img.reshape(c, h * w)
            perm_flat = flat[:, self.perm]
            return perm_flat.reshape(c, h, w)
        else:
            flat = img.flatten()
            perm_flat = flat[self.perm]
            return perm_flat.reshape(3, 32, 32)

class PatchPermutationTransform:
    """
    Divides the 32x32 image into a grid of PxP patches (e.g. 4x4 grid of 8x8 patches)
    and shuffles the patches according to a fixed permutation.
    Internal spatial correlation inside each patch is 100% preserved.
    """
    def __init__(self, patch_size=8, seed=26013):
        self.patch_size = patch_size
        self.seed = seed
        self.num_patches = (32 // patch_size) ** 2
        rng = np.random.RandomState(seed)
        self.perm = rng.permutation(self.num_patches)

    def __call__(self, img):
        # img: (3, 32, 32)
        c, h, w = img.shape
        p = self.patch_size
        gh = h // p
        gw = w // p
        
        # Deconstruct into patches: (c, gh, p, gw, p) -> (gh, gw, c, p, p)
        patches = img.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(self.num_patches, c, p, p)
        
        # Permute patches
        perm_patches = patches[self.perm]
        
        # Reconstruct back to (c, 32, 32)
        reconstructed = perm_patches.reshape(gh, gw, c, p, p).transpose(2, 0, 3, 1, 4).reshape(c, h, w)
        return reconstructed

class GaussianNoiseTransform:
    """
    Adds zero-mean Gaussian noise N(0, sigma^2) and clips to [0, 1].
    """
    def __init__(self, sigma=0.2, seed=None):
        self.sigma = sigma
        self.seed = seed

    def __call__(self, img):
        # img: (3, 32, 32) float32 in [0, 1]
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        noise = rng.normal(0, self.sigma, img.shape).astype(np.float32)
        corrupted = np.clip(img + noise, 0.0, 1.0)
        return corrupted
