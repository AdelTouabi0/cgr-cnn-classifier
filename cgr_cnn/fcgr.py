"""Images FCGR (Frequency Chaos Game Representation) de fragments d'ADN.

Version autonome et vectorisée du module FCGR du projet
``cgr-genomic-signatures`` : une séquence devient une image 2^k x 2^k dont
chaque pixel est la fréquence d'un k-mer.
"""

from functools import cache
from itertools import product

import numpy as np

BASES = "ACGT"
# Coins CGR : A en haut à gauche, T en haut à droite, C en bas à gauche,
# G en bas à droite.
CORNERS = {"A": (0.0, 1.0), "T": (1.0, 1.0), "C": (0.0, 0.0), "G": (1.0, 0.0)}
_ENCODING = np.full(256, -1, dtype=np.int64)
for _i, _base in enumerate(BASES):
    _ENCODING[ord(_base)] = _i


def kmer_cell(kmer):
    """Renvoie la case (ligne, colonne) d'un k-mer dans l'image FCGR."""
    size = 2 ** len(kmer)
    x, y = 0.5, 0.5
    for base in kmer:
        cx, cy = CORNERS[base]
        x, y = (x + cx) / 2, (y + cy) / 2
    return min(int((1 - y) * size), size - 1), min(int(x * size), size - 1)


@cache
def _flat_index(k):
    """Table code entier du k-mer -> indice du pixel dans l'image aplatie."""
    size = 2 ** k
    index = np.empty(4 ** k, dtype=np.int64)
    for code, letters in enumerate(product(BASES, repeat=k)):
        row, col = kmer_cell("".join(letters))
        index[code] = row * size + col
    return index


def fcgr_image(sequence, k=6):
    """Renvoie l'image FCGR normalisée (float32, somme à 1) d'une séquence.

    Les k-mers contenant une base ambiguë sont ignorés.
    """
    codes = _ENCODING[np.frombuffer(sequence.upper().encode("ascii"),
                                    dtype=np.uint8)]
    n = len(codes) - k + 1
    size = 2 ** k
    image = np.zeros(size * size, dtype=np.float32)
    if n <= 0:
        return image.reshape(size, size)
    kmer_codes = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for i in range(k):
        window = codes[i:i + n]
        kmer_codes = kmer_codes * 4 + window
        valid &= window >= 0
    counts = np.bincount(kmer_codes[valid], minlength=4 ** k)
    image[_flat_index(k)] = counts
    total = image.sum()
    if total > 0:
        image /= total
    return image.reshape(size, size)
