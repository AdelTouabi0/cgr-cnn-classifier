"""Construction des jeux de données : fragments d'ADN -> images FCGR.

Deux sources sont possibles :
- ``load_genome_folders`` : vrais génomes, un sous-dossier par classe ;
- ``simulate_markov_genomes`` : génomes simulés par chaînes de Markov, pour
  valider la méthode sans téléchargement.
"""

from pathlib import Path

import numpy as np
from Bio import SeqIO

from .fcgr import fcgr_image

FORMATS = {".fa": "fasta", ".fasta": "fasta", ".fna": "fasta",
           ".gb": "genbank", ".gbk": "genbank"}


def load_genome_folders(root):
    """Lit ``root/<classe>/*.fasta|gb`` et renvoie {classe: [séquences]}."""
    genomes = {}
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir():
            continue
        sequences = []
        for path in sorted(class_dir.iterdir()):
            fmt = FORMATS.get(path.suffix.lower())
            if fmt:
                sequences.extend(str(rec.seq).upper()
                                 for rec in SeqIO.parse(path, fmt))
        if sequences:
            genomes[class_dir.name] = sequences
    if len(genomes) < 2:
        raise ValueError(f"Il faut au moins 2 classes dans {root}")
    return genomes


def simulate_markov_genomes(n_classes=5, length=200_000, order=3,
                            divergence=0.15, seed=0):
    """Simule un génome par classe avec des chaînes de Markov d'ordre ``order``.

    Toutes les classes partent d'une même matrice de transition « ancestrale »
    à laquelle on ajoute un bruit d'amplitude ``divergence`` : plus elle est
    faible, plus les classes se ressemblent et plus la tâche est difficile.
    """
    rng = np.random.default_rng(seed)
    n_contexts = 4 ** order
    ancestor = rng.dirichlet(np.full(4, 2.0), size=n_contexts)
    genomes = {}
    for c in range(n_classes):
        noise = rng.dirichlet(np.full(4, 2.0), size=n_contexts)
        transitions = (1 - divergence) * ancestor + divergence * noise
        cumulative = transitions.cumsum(axis=1)
        draws = rng.random(length)
        codes = np.empty(length, dtype=np.int64)
        codes[:order] = rng.integers(0, 4, order)
        context = 0
        for i in range(order):
            context = context * 4 + codes[i]
        for i in range(order, length):
            base = int(np.searchsorted(cumulative[context], draws[i]))
            codes[i] = min(base, 3)
            context = (context * 4 + codes[i]) % n_contexts
        genomes[f"classe_{c + 1}"] = ["".join(np.array(list("ACGT"))[codes])]
    return genomes


def sample_fragments(genomes, fragment_length, n_per_class, seed=0):
    """Tire ``n_per_class`` fragments aléatoires dans les génomes de chaque classe.

    Renvoie la liste des fragments, leurs étiquettes entières et la liste
    ordonnée des noms de classes.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(genomes)
    fragments, labels = [], []
    for label, name in enumerate(classes):
        sequences = [s for s in genomes[name] if len(s) >= fragment_length]
        if not sequences:
            raise ValueError(f"Aucune séquence assez longue pour {name}")
        lengths = np.array([len(s) for s in sequences], dtype=float)
        for _ in range(n_per_class):
            # Choix d'une séquence proportionnellement à sa longueur.
            seq = sequences[rng.choice(len(sequences), p=lengths / lengths.sum())]
            start = rng.integers(0, len(seq) - fragment_length + 1)
            fragments.append(seq[start:start + fragment_length])
            labels.append(label)
    return fragments, np.array(labels), classes


def to_images(fragments, k=6):
    """Convertit des fragments en tableau d'images FCGR (n, 1, 2^k, 2^k)."""
    images = np.stack([fcgr_image(f, k) for f in fragments])
    return images[:, None, :, :]
