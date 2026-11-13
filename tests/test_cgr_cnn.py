"""Tests rapides : images FCGR, données, modèle et pipeline complète."""

import json

import numpy as np
import pytest
import torch

from cgr_cnn.data import sample_fragments, simulate_markov_genomes, to_images
from cgr_cnn.fcgr import fcgr_image, kmer_cell
from cgr_cnn.model import FCGRNet, predict
from cgr_cnn.train import main


def test_fcgr_image_shape_and_normalization():
    """L'image fait 2^k x 2^k et somme à 1."""
    image = fcgr_image("ACGT" * 100, k=4)
    assert image.shape == (16, 16)
    assert image.sum() == pytest.approx(1.0)


def test_fcgr_pixel_matches_kmer():
    """Une séquence homopolymère n'allume que la case de son k-mer."""
    image = fcgr_image("A" * 50, k=3)
    assert image[kmer_cell("AAA")] == pytest.approx(1.0)


def test_fcgr_ignores_ambiguous_bases():
    """Une séquence de N donne une image vide."""
    assert fcgr_image("N" * 30, k=3).sum() == 0


def test_simulation_is_reproducible():
    """La même graine donne les mêmes génomes."""
    a = simulate_markov_genomes(n_classes=2, length=2000, seed=1)
    b = simulate_markov_genomes(n_classes=2, length=2000, seed=1)
    assert a == b
    assert set(a["classe_1"][0]) <= set("ACGT")


def test_sample_fragments_balanced():
    """Autant de fragments par classe, de la bonne longueur."""
    genomes = simulate_markov_genomes(n_classes=3, length=5000, seed=0)
    fragments, labels, classes = sample_fragments(genomes, 300, 10)
    assert classes == ["classe_1", "classe_2", "classe_3"]
    assert np.bincount(labels).tolist() == [10, 10, 10]
    assert {len(f) for f in fragments} == {300}


def test_model_output_shape():
    """Le CNN renvoie un logit par classe."""
    model = FCGRNet(n_classes=4, image_size=32)
    images = to_images(["ACGT" * 50] * 3, k=5).astype(np.float32)
    assert model(torch.as_tensor(images)).shape == (3, 4)
    assert predict(model, images).shape == (3,)


def test_pipeline_runs_end_to_end(tmp_path):
    """La pipeline complète tourne sur un mini jeu simulé."""
    main(["--simulate", "-k", "4", "--fragment-length", "300",
          "--n-per-class", "40", "--epochs", "2", "-o", str(tmp_path)])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert 0 <= metrics["cnn_accuracy"] <= 1
    assert (tmp_path / "confusion_cnn.png").exists()
