"""Orquestador de FedAvg para embeddings Word2Vec de hogares IoT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clients.client import evaluate_client, fit
from shared.model import EMBEDDING_DIM, nearest_words, initialize_embeddings

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLIENTS_DIR = PROJECT_DIR / "clients"


def discover_clients() -> list[Path]:
    return sorted(
        path
        for path in CLIENTS_DIR.glob("casa_*")
        if (path / "data" / "train.txt").is_file()
        and (path / "data" / "test.txt").is_file()
    )


def aggregate_metrics(
    client_dirs: list[Path],
    vocabulary: list[str],
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
) -> float:
    total_loss = 0.0
    total_pairs = 0
    for client_index, client_dir in enumerate(client_dirs):
        loss, pair_count = evaluate_client(
            client_dir,
            vocabulary,
            input_vectors,
            output_vectors,
            seed=9000 + client_index,
        )
        total_loss += loss * pair_count
        total_pairs += pair_count
    if total_pairs == 0:
        raise ValueError("No hay pares de palabras para evaluar")
    return total_loss / total_pairs


def run(rounds: int, local_epochs: int, learning_rate: float) -> tuple[np.ndarray, np.ndarray]:
    client_dirs = discover_clients()
    if not client_dirs:
        raise FileNotFoundError(
            "No se encontraron corpus de clientes. Ejecuta primero: python generate_data.py"
        )
    vocabulary_path = PROJECT_DIR / "shared" / "vocabulary.json"
    if not vocabulary_path.is_file():
        raise FileNotFoundError("Falta el vocabulario. Ejecuta: python generate_data.py")
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    input_vectors, output_vectors = initialize_embeddings(
        len(vocabulary), EMBEDDING_DIM
    )
    loss = aggregate_metrics(client_dirs, vocabulary, input_vectors, output_vectors)
    print(f"Clientes: {len(client_dirs)} | perdida inicial skip-gram: {loss:.4f}")

    for round_number in range(1, rounds + 1):
        client_updates: list[tuple[np.ndarray, np.ndarray, int]] = []
        for client_index, client_dir in enumerate(client_dirs):
            updated_input, updated_output, pair_count = fit(
                client_dir,
                vocabulary,
                input_vectors,
                output_vectors,
                local_epochs,
                learning_rate,
                seed=round_number * 1000 + client_index,
            )
            client_updates.append((updated_input, updated_output, pair_count))

        total_pairs = sum(pair_count for _, _, pair_count in client_updates)
        input_vectors = sum(
            updated_input * pair_count
            for updated_input, _, pair_count in client_updates
        ) / total_pairs
        output_vectors = sum(
            updated_output * pair_count
            for _, updated_output, pair_count in client_updates
        ) / total_pairs
        loss = aggregate_metrics(
            client_dirs, vocabulary, input_vectors, output_vectors
        )
        print(
            f"Ronda {round_number:02d}/{rounds:02d} | "
            f"perdida skip-gram: {loss:.4f} | pares agregados: {total_pairs}"
        )

    artifact_path = PROJECT_DIR / "server" / "artifacts" / "global_model.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "algorithm": "word2vec_skip_gram_negative_sampling",
                "embedding_dim": EMBEDDING_DIM,
                "vocabulary": vocabulary,
                "input_vectors": input_vectors.tolist(),
                "output_vectors": output_vectors.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Modelo global guardado en: {artifact_path.relative_to(PROJECT_DIR)}")
    for word in ("consumo_alto", "clima_encendido", "hogar_ocupado"):
        neighbors = nearest_words(word, vocabulary, input_vectors)
        if neighbors:
            formatted = ", ".join(f"{token} ({score:.2f})" for token, score in neighbors)
            print(f"Palabras cercanas a {word}: {formatted}")
    return input_vectors, output_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    args = parser.parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.learning_rate <= 0:
        parser.error("rounds, local-epochs y learning-rate deben ser positivos")
    run(args.rounds, args.local_epochs, args.learning_rate)


if __name__ == "__main__":
    main()