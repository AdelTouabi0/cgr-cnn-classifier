"""Télécharge 5 génomes bactériens de référence depuis NCBI (FASTA).

Usage :
    python scripts/download_bacteria.py --email prenom.nom@exemple.fr

Crée data/real/<espèce>/<accession>.fasta, la structure attendue par
``python -m cgr_cnn.train --data data/real``.
"""

import argparse
import time
from pathlib import Path

from Bio import Entrez

# Génomes complets de souches de référence (RefSeq).
GENOMES = {
    "Escherichia_coli": "NC_000913",          # K-12 MG1655
    "Bacillus_subtilis": "NC_000964",         # souche 168
    "Staphylococcus_aureus": "NC_007795",     # NCTC 8325
    "Mycobacterium_tuberculosis": "NC_000962",  # H37Rv
    "Pseudomonas_aeruginosa": "NC_002516",    # PAO1
}


def main():
    """Télécharge chaque génome manquant."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True,
                        help="Adresse e-mail transmise à NCBI Entrez")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/real"))
    args = parser.parse_args()
    Entrez.email = args.email

    for species, accession in GENOMES.items():
        folder = args.output / species
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{accession}.fasta"
        if path.exists():
            print(f"{species} : déjà présent")
            continue
        with Entrez.efetch(db="nuccore", id=accession, rettype="fasta",
                           retmode="text") as handle:
            path.write_text(handle.read(), encoding="utf-8")
        header = path.read_text(encoding="utf-8").splitlines()[0]
        print(f"{species} : {header[:80]}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
