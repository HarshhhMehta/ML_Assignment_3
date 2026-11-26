import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with symmetrical encoder/decoder architecture.
    The architecture differs strictly in the bottleneck dimension:
    - Small: 64
    - Medium: 256
    - Large: 1024
    """
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        
        # Symmetrical Encoder Backbone
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2
            nn.ReLU(inplace=True)
        )
        self.enc_fc = nn.Linear(64 * 2 * 2, bottleneck_dim)
        
        # Symmetrical Decoder Backbone
        self.dec_fc = nn.Linear(bottleneck_dim, 64 * 2 * 2)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 2x2 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1), # 4x4 -> 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Sigmoid() # Bound outputs in [0, 1] for unit-normalized images
        )

    def encode(self, x):
        """Extracts the bottleneck representation."""
        h = self.enc_conv(x)
        b = self.enc_fc(h.view(h.size(0), -1))
        return b

    def decode(self, b):
        """Reconstructs the image from the bottleneck representation."""
        d = nn.functional.relu(self.dec_fc(b)).view(-1, 64, 2, 2)
        out = self.dec_conv(d)
        return out

    def forward(self, x):
        b = self.encode(x)
        out = self.decode(b)
        return out

def get_ae_parameter_counts(model):
    enc_params = sum(p.numel() for p in list(model.enc_conv.parameters()) + list(model.enc_fc.parameters()))
    dec_params = sum(p.numel() for p in list(model.dec_conv.parameters()) + list(model.dec_fc.parameters()))
    return {
        'encoder_params': enc_params,
        'decoder_params': dec_params,
        'total_params': enc_params + dec_params
    }
