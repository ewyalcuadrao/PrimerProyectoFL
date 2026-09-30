"""Word2Vec Skip-Gram con muestreo negativo para tokens IoT."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMBEDDING_DIM = 24
CONTEXT_WINDOW = 2
NEGATIVE_SAMPLES = 4
BATCH_SIZE = 512


def load_sentences(path: Path, vocabulary: list[str]) -> list[np.ndarray]:
    """Convierte cada linea del corpus en una secuencia de ids de palabras."""
    word_ids = {word: index for index, word in enumerate(vocabulary)}
    sentences: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = [word_ids[word] for word in line.split() if word in word_ids]
        if tokens:
            sentences.append(np.asarray(tokens, dtype=np.int64))
    return sentences


def initialize_embeddings(
    vocabulary_size: int, embedding_dim: int = EMBEDDING_DIM, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Crea la misma base de embeddings para todos los clientes federados."""
    rng = np.random.default_rng(seed)
    input_vectors = rng.uniform(
        -0.5 / embedding_dim, 0.5 / embedding_dim, (vocabulary_size, embedding_dim)
    )
    output_vectors = np.zeros((vocabulary_size, embedding_dim), dtype=np.float64)
    return input_vectors, output_vectors


def _context_pairs(sentences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []
    for sentence in sentences:
        for position, center in enumerate(sentence):
            start = max(0, position - CONTEXT_WINDOW)
            end = min(len(sentence), position + CONTEXT_WINDOW + 1)
            for context_position in range(start, end):
                if context_position != position:
                    centers.append(int(center))
                    contexts.append(int(sentence[context_position]))
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def _negative_distribution(
    sentences: list[np.ndarray], vocabulary_size: int
) -> np.ndarray:
    token_counts = np.zeros(vocabulary_size, dtype=np.float64)
    for sentence in sentences:
        token_counts += np.bincount(sentence, minlength=vocabulary_size)
    smoothed_counts = np.maximum(token_counts, 1e-3) ** 0.75
    return smoothed_counts / smoothed_counts.sum()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


def train_local(
    sentences: list[np.ndarray],
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Entrena localmente pares palabra-contexto sin enviar el corpus."""
    centers, contexts = _context_pairs(sentences)
    if len(centers) == 0:
        raise ValueError("El corpus local no contiene pares de contexto")

    local_input = input_vectors.copy()
    local_output = output_vectors.copy()
    rng = np.random.default_rng(seed)
    distribution = _negative_distribution(sentences, len(local_input))
    samples_per_center = NEGATIVE_SAMPLES + 1

    for _ in range(epochs):
        order = rng.permutation(len(centers))
        for batch_start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[batch_start : batch_start + BATCH_SIZE]
            batch_centers = centers[batch_indices]
            positive_contexts = contexts[batch_indices]
            negative_contexts = rng.choice(
                len(local_input),
                size=(len(batch_indices), NEGATIVE_SAMPLES),
                p=distribution,
            )
            target_ids = np.column_stack((positive_contexts, negative_contexts))
            repeated_centers = np.repeat(batch_centers, samples_per_center)
            flat_targets = target_ids.reshape(-1)
            center_vectors = local_input[repeated_centers]
            target_vectors = local_output[flat_targets]
            scores = np.sum(center_vectors * target_vectors, axis=1)
            labels = np.zeros(len(scores), dtype=np.float64)
            labels[::samples_per_center] = 1.0
            score_gradients = (labels - _sigmoid(scores)) * (
                learning_rate / len(batch_indices)
            )
            input_gradients = (
                score_gradients[:, None] * target_vectors
            ).reshape(len(batch_indices), samples_per_center, -1).sum(axis=1)
            output_gradients = score_gradients[:, None] * center_vectors
            np.add.at(local_input, batch_centers, input_gradients)
            np.add.at(local_output, flat_targets, output_gradients)

    return local_input, local_output, len(centers)


def evaluate_local(
    sentences: list[np.ndarray],
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
    seed: int,
) -> tuple[float, int]:
    """Mide la perdida skip-gram en pares de palabras no usados para entrenar."""
    centers, contexts = _context_pairs(sentences)
    if len(centers) == 0:
        raise ValueError("El corpus de prueba no contiene pares de contexto")
    rng = np.random.default_rng(seed)
    distribution = _negative_distribution(sentences, len(input_vectors))
    losses: list[np.ndarray] = []
    for batch_start in range(0, len(centers), BATCH_SIZE):
        batch_centers = centers[batch_start : batch_start + BATCH_SIZE]
        positive_contexts = contexts[batch_start : batch_start + BATCH_SIZE]
        negative_contexts = rng.choice(
            len(input_vectors),
            size=(len(batch_centers), NEGATIVE_SAMPLES),
            p=distribution,
        )
        target_ids = np.column_stack((positive_contexts, negative_contexts))
        scores = np.sum(
            input_vectors[batch_centers, None, :] * output_vectors[target_ids], axis=2
        )
        positive_loss = np.logaddexp(0.0, -scores[:, 0])
        negative_loss = np.logaddexp(0.0, scores[:, 1:]).sum(axis=1)
        losses.append(positive_loss + negative_loss)
    return float(np.concatenate(losses).mean()), len(centers)


def nearest_words(
    word: str, vocabulary: list[str], input_vectors: np.ndarray, count: int = 5
) -> list[tuple[str, float]]:
    if word not in vocabulary:
        return []
    norms = np.linalg.norm(input_vectors, axis=1)
    normalized = input_vectors / np.maximum(norms[:, None], 1e-12)
    similarities = normalized @ normalized[vocabulary.index(word)]
    similarities[vocabulary.index(word)] = -np.inf
    nearest_ids = np.argsort(similarities)[-count:][::-1]
    return [(vocabulary[index], float(similarities[index])) for index in nearest_ids]