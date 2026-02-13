import torch  # pyright: ignore[reportUnusedImport, reportUnusedImport, reportUnusedImport, reportUnusedImport, reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

class EGCAutoencoder(nn.Module):
    def __init__(self, window_size=360):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # between layers
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, window_size),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed