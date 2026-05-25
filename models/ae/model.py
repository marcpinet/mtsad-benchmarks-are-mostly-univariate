import torch.nn as nn
import torch.nn.functional as F


class AECI(nn.Module):
    def __init__(self, window_size, latent_dim, n_channels):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        hidden = max(latent_dim * 2, window_size // 2)
        self.enc1 = nn.Linear(window_size, hidden)
        self.enc2 = nn.Linear(hidden, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden)
        self.dec2 = nn.Linear(hidden, window_size)
    def forward(self, x):
        B, T, C = x.shape
        h = x.permute(0, 2, 1).reshape(B * C, T)
        r = self.dec2(F.gelu(self.dec1(F.gelu(self.enc2(F.gelu(self.enc1(h))))))).reshape(B, C, T).permute(0, 2, 1)
        return x, r


class AECD(nn.Module):
    def __init__(self, window_size, latent_dim, n_channels):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        flat = window_size * n_channels
        hidden = max(latent_dim * 2, flat // 2)
        self.enc1 = nn.Linear(flat, hidden)
        self.enc2 = nn.Linear(hidden, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden)
        self.dec2 = nn.Linear(hidden, flat)
    def forward(self, x):
        B = x.size(0)
        r = self.dec2(F.gelu(self.dec1(F.gelu(self.enc2(F.gelu(self.enc1(x.reshape(B, -1)))))))).reshape(B, self.window_size, self.n_channels)
        return x, r