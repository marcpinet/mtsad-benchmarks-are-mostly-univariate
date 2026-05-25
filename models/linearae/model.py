import torch.nn as nn


class LinearAECI(nn.Module):
    def __init__(self, window_size, latent_dim, n_channels):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.enc = nn.Linear(window_size, latent_dim)
        self.dec = nn.Linear(latent_dim, window_size)

    def forward(self, x):
        B, T, C = x.shape
        h = x.permute(0, 2, 1).reshape(B * C, T)
        r = self.dec(self.enc(h)).reshape(B, C, T).permute(0, 2, 1)
        return x, r


class LinearAECD(nn.Module):
    def __init__(self, window_size, latent_dim, n_channels):
        super().__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        flat = window_size * n_channels
        self.enc = nn.Linear(flat, latent_dim)
        self.dec = nn.Linear(latent_dim, flat)

    def forward(self, x):
        B = x.size(0)
        r = self.dec(self.enc(x.reshape(B, -1))).reshape(B, self.window_size, self.n_channels)
        return x, r
