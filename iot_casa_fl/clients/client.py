"""Entrenamiento Word2Vec ejecutado localmente por cada hogar."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.model import evaluate_local, load_sentences, train_local


def fit(
    client_dir: Path,
    vocabulary: list[str],
    global_input: np.ndarray,
    global_output: np.ndarray,
    local_epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    sentences = load_sentences(client_dir / "data" / "train.txt", vocabulary)
    updated_input, updated_output, pair_count = train_local(
        sentences,
        global_input,
        global_output,
        local_epochs,
        learning_rate,
        seed,
    )
    return updated_input, updated_output, pair_count


def evaluate_client(
    client_dir: Path,
    vocabulary: list[str],
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
    seed: int,
) -> tuple[float, int]:
    sentences = load_sentences(client_dir / "data" / "test.txt", vocabulary)
    return evaluate_local(sentences, input_vectors, output_vectors, seed)