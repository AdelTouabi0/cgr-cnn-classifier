"""Réseau de neurones convolutif pour images FCGR."""

import torch
from torch import nn


class FCGRNet(nn.Module):
    """Petit CNN : trois blocs convolutifs puis un classifieur dense.

    Dans une image FCGR, les k-mers qui partagent leur suffixe sont voisins :
    les convolutions exploitent cette localité. En revanche, la position d'un
    pixel désigne un k-mer précis. Contrairement aux CNN pour photos, on
    n'utilise donc pas de pooling global (qui rendrait le modèle invariant à
    la position et effacerait l'identité des k-mers) : les cartes de
    caractéristiques sont aplaties avant le classifieur.
    """

    def __init__(self, n_classes, image_size=64, channels=(16, 32, 64),
                 hidden=128, dropout=0.3):
        """Construit le réseau pour des images ``image_size`` x ``image_size``."""
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.AvgPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        reduced = image_size // 2 ** len(channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * reduced * reduced, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        """Renvoie les logits (n, n_classes) pour un lot d'images."""
        return self.classifier(self.features(x))


def standardize(images, mean=None, std=None):
    """Centre-réduit les images (statistiques calculées sur l'entraînement)."""
    if mean is None:
        mean, std = images.mean(), images.std()
    return (images - mean) / (std + 1e-8), mean, std


def count_parameters(model):
    """Renvoie le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, images, batch_size=256):
    """Renvoie les classes prédites pour un tableau d'images NumPy."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(images[start:start + batch_size])
            predictions.append(model(batch).argmax(dim=1))
    return torch.cat(predictions).numpy()
