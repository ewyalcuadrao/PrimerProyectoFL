"""Red de ejemplo y entrenamiento federado sobre secuencias de Interests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ndn_sim.network import FibEntry, Forwarder, ForwardingResult, Network, NoRouteError
from shared.model import EMBEDDING_DIM, initialize_embeddings, nearest_words, train_local

PROJECT_DIR = Path(__file__).resolve().parents[1]

""" Nombres de Interests de ejemplo para tres bordes, un core y cuatro productores.
Los patrones de tráfico locales se repiten varias veces para simular sesiones de usuario.
El modelo Word2Vec se entrena localmente en cada borde y luego se agregan los vectores de embedding en el core para producir un modelo global.
El modelo global se utiliza para predecir Interests conocidos a partir del contexto local reciente, y se compara con el forwarding normal basado en
 FIB. Se generan métricas de aciertos, falsos positivos y bloqueos seguros, y se guarda un informe detallado en CSV."""


# Nombres de Interests de ejemplo para tres bordes, un core y cuatro productores.
VIDEO_1 = "/ndn/video/canal1/segment/001"
VIDEO_2 = "/ndn/video/canal1/segment/002"
VIDEO_3 = "/ndn/video/canal2/segment/001"
GUIDE_1 = "/ndn/guide/canal1/now"
GUIDE_2 = "/ndn/guide/canal2/now"
NEWS_CENTRO = "/ndn/news/region/centro/item/001"
NEWS_NORTE = "/ndn/news/region/norte/item/002"
WEATHER_CENTRO = "/ndn/weather/region/centro/forecast/today"
WEATHER_NORTE = "/ndn/weather/region/norte/forecast/today"
WEATHER_SUR = "/ndn/weather/region/sur/forecast/today"

LOCAL_PATTERNS: dict[str, list[list[str]]] = {
    "edge_centro": [
        [VIDEO_1, GUIDE_1, VIDEO_2, NEWS_CENTRO],
        [VIDEO_2, GUIDE_1, VIDEO_1, WEATHER_CENTRO],
        [VIDEO_1, GUIDE_1, VIDEO_2],
    ],
    "edge_norte": [
        [WEATHER_NORTE, NEWS_NORTE, WEATHER_NORTE, VIDEO_3],
        [NEWS_NORTE, WEATHER_NORTE, VIDEO_3, GUIDE_2],
        [WEATHER_NORTE, NEWS_NORTE, WEATHER_NORTE],
    ],
    "edge_sur": [
        [VIDEO_1, GUIDE_1, VIDEO_2, WEATHER_SUR],
        [WEATHER_SUR, NEWS_CENTRO, WEATHER_SUR, VIDEO_1],
        [VIDEO_2, GUIDE_1, VIDEO_1],
        [GUIDE_1, VIDEO_2, WEATHER_SUR],
    ],
}

# Parámetros para la inferencia de Interests
SIMILARITY_THRESHOLD = 0.30
COMPARISON_REQUESTS = 100

# Clase para representar una prueba de Interest
@dataclass(frozen=True)
class InterestProbe:
    node: str
    name: str
    context: tuple[str, ...]
    expected_target: str | None

# Funciones para construir la red, generar sesiones locales, entrenar el modelo Word2Vec y comparar políticas de forwarding.
def build_network() -> Network:
    """Crea tres bordes, un core con rutas agregadas y cuatro productores."""
    video_channel1 = {VIDEO_1, VIDEO_2}
    video_other = {VIDEO_3}
    all_interests = (
        video_channel1
        | video_other
        | {GUIDE_1, GUIDE_2, NEWS_CENTRO, NEWS_NORTE}
        | {WEATHER_CENTRO, WEATHER_NORTE, WEATHER_SUR}
    )

    forwarders = {
        edge_name: Forwarder(edge_name, [FibEntry("/ndn", "core")])
        for edge_name in LOCAL_PATTERNS
    }
    forwarders.update(
        {
            "core": Forwarder(
                "core",
                [
                    FibEntry("/ndn/video", "producer_video"),
                    FibEntry("/ndn/video/canal1", "producer_canal1"),
                    FibEntry("/ndn/guide", "producer_guide"),
                    FibEntry("/ndn/news", "producer_news"),
                    FibEntry("/ndn/weather", "producer_weather"),
                ],
            ),
            "producer_video": Forwarder("producer_video", content_store=video_other),
            "producer_canal1": Forwarder(
                "producer_canal1", content_store=video_channel1
            ),
            "producer_guide": Forwarder(
                "producer_guide", content_store={GUIDE_1, GUIDE_2}
            ),
            "producer_news": Forwarder(
                "producer_news", content_store={NEWS_CENTRO, NEWS_NORTE}
            ),
            "producer_weather": Forwarder(
                "producer_weather",
                content_store={WEATHER_CENTRO, WEATHER_NORTE, WEATHER_SUR},
            ),
        }
    )
    network = Network(forwarders)
    for forwarder in forwarders.values():
        for entry in forwarder.fib:
            if entry.prefix in all_interests:
                raise ValueError(f"La FIB contiene un Interest exacto: {entry.prefix}")
    return network


def collect_local_sessions(
    network: Network, repetitions: int = 16
) -> tuple[dict[str, list[list[str]]], dict[str, ForwardingResult]]:
    """Genera trafico de muestra y conserva cada secuencia en su nodo de borde."""
    local_sessions: dict[str, list[list[str]]] = {}
    examples: dict[str, ForwardingResult] = {}
    for edge_name, patterns in LOCAL_PATTERNS.items():
        sessions: list[list[str]] = []
        for _ in range(repetitions):
            for pattern in patterns:
                session: list[str] = []
                for interest_name in pattern:
                    result = network.forward_interest(edge_name, interest_name)
                    examples.setdefault(interest_name, result)
                    session.append(interest_name)
                sessions.append(session)
        local_sessions[edge_name] = sessions
    return local_sessions, examples


def _to_word_ids(
    sessions: dict[str, list[list[str]]], vocabulary: list[str]
) -> dict[str, list[np.ndarray]]:
    word_ids = {word: index for index, word in enumerate(vocabulary)}
    return {
        client_id: [
            np.asarray([word_ids[name] for name in session], dtype=np.int64)
            for session in client_sessions
        ]
        for client_id, client_sessions in sessions.items()
    }


def infer_interest_from_context(
    context: tuple[str, ...],
    vocabulary: list[str],
    input_vectors: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str | None, float]:
    """Predice un Interest conocido desde el contexto local reciente."""
    word_ids = {word: index for index, word in enumerate(vocabulary)}
    context_ids = [word_ids[word] for word in context if word in word_ids]
    if not context_ids:
        return None, 0.0

    context_vector = input_vectors[context_ids].mean(axis=0)
    context_norm = np.linalg.norm(context_vector)
    candidate_norms = np.linalg.norm(input_vectors, axis=1)
    denominators = context_norm * candidate_norms
    similarities = np.divide(
        input_vectors @ context_vector,
        denominators,
        out=np.zeros(len(vocabulary), dtype=np.float64),
        where=denominators > 1e-12,
    )
    similarities[context_ids] = -np.inf
    best_index = int(np.argmax(similarities))
    confidence = float(similarities[best_index])
    if confidence < threshold:
        return None, confidence
    return vocabulary[best_index], confidence


def make_comparison_probes(total: int = COMPARISON_REQUESTS) -> list[InterestProbe]:
    """Crea 80 nombres exactos y 20 OOV con positivos y negativos conocidos."""
    if total != 100:
        raise ValueError("La comparativa de referencia requiere exactamente 100 Interests")

    edge_names = list(LOCAL_PATTERNS)
    streams = {
        edge_name: [name for pattern in patterns for name in pattern]
        for edge_name, patterns in LOCAL_PATTERNS.items()
    }
    probes: list[InterestProbe] = []
    for index in range(80):
        edge_name = edge_names[index % len(edge_names)]
        stream = streams[edge_name]
        position = (index // len(edge_names)) % len(stream)
        context = tuple(stream[max(0, position - 2) : position])
        name = stream[position]
        probes.append(InterestProbe(edge_name, name, context, name))

    correlated_context = (GUIDE_1, VIDEO_2)
    for index in range(20):
        edge_name = edge_names[index % len(edge_names)]
        name = (
            f"/ndn/weather/region/sur/forecast/alias/{index:03d}"
            if index < 10
            else f"/ndn/unmodeled/request/alias/{index:03d}"
        )
        if index < 10:
            expected_target = WEATHER_SUR
            context = correlated_context
        elif index < 15:
            expected_target = None
            context = correlated_context
        else:
            expected_target = None
            context = (f"/ndn/unseen/device/event/{index:03d}",)
        probes.append(InterestProbe(edge_name, name, context, expected_target))
    return probes


def compare_forwarding_policies(
    probes: list[InterestProbe],
    vocabulary: list[str],
    input_vectors: np.ndarray,
    output_vectors: np.ndarray,
) -> tuple[dict[str, dict[str, int]], list[dict[str, str]]]:
    """Compara forwarding normal y forwarding asistido por Word2Vec."""
    summaries = {
        "baseline": {
            "correct_data": 0,
            "false_positives": 0,
            "valid_missed": 0,
            "safely_blocked": 0,
            "no_data": 0,
            "below_threshold": 0,
            "wrong_destination": 0,
        },
        "word2vec": {
            "correct_data": 0,
            "false_positives": 0,
            "valid_missed": 0,
            "safely_blocked": 0,
            "no_data": 0,
            "below_threshold": 0,
            "wrong_destination": 0,
        },
    }
    report_rows: list[dict[str, str]] = []
    known_names = set(vocabulary)

    for request_id, probe in enumerate(probes, start=1):
        is_exact = probe.name in known_names
        prediction, confidence = (probe.name, 1.0) if is_exact else infer_interest_from_context(
            probe.context,
            vocabulary,
            input_vectors,
        )
        policy_results: dict[str, tuple[str, str]] = {}
        for policy in ("baseline", "word2vec"):
            summary = summaries[policy]
            sent_name = probe.name if policy == "baseline" else prediction
            if policy == "baseline" and is_exact:
                sent_name = probe.name
            forwarded_to_data = False
            if sent_name is not None:
                try:
                    result = build_network().forward_interest(probe.node, sent_name)
                    forwarded_to_data = result.satisfied_by.startswith("producer_")
                except NoRouteError:
                    forwarded_to_data = False

            outcome = "exact" if is_exact else "no_data"
            if not is_exact:
                if sent_name is None:
                    summary["below_threshold"] += 1
                    if probe.expected_target is None:
                        summary["safely_blocked"] += 1
                        outcome = "blocked_no_context" if confidence <= 0 else "blocked_below_threshold"
                    else:
                        summary["valid_missed"] += 1
                        outcome = "valid_target_missed"
                elif not forwarded_to_data:
                    summary["no_data"] += 1
                    if probe.expected_target is not None:
                        summary["valid_missed"] += 1
                    outcome = "no_data"
                elif probe.expected_target is None:
                    summary["false_positives"] += 1
                    outcome = "false_positive"
                elif sent_name == probe.expected_target:
                    summary["correct_data"] += 1
                    outcome = "recovered_correct_target"
                else:
                    summary["false_positives"] += 1
                    summary["valid_missed"] += 1
                    summary["wrong_destination"] += 1
                    outcome = "wrong_destination"
            elif forwarded_to_data:
                summary["correct_data"] += 1
            else:
                summary["no_data"] += 1
                outcome = "no_data"
            policy_results[policy] = (sent_name or "", outcome)

        report_rows.append(
            {
                "request_id": str(request_id),
                "node": probe.node,
                "interest_received": probe.name,
                "context": " | ".join(probe.context),
                "expected_target": probe.expected_target or "",
                "baseline_forwarded": policy_results["baseline"][0],
                "baseline_outcome": policy_results["baseline"][1],
                "word2vec_forwarded": policy_results["word2vec"][0],
                "word2vec_outcome": policy_results["word2vec"][1],
                "cosine_similarity": f"{confidence:.4f}",
            }
        )

    return summaries, report_rows


def print_global_relations(
    vocabulary: list[str], input_vectors: np.ndarray, count: int = 3
) -> None:
    """Muestra y exporta los vecinos Word2Vec del modelo global entrenado."""
    relations: list[dict[str, str]] = []
    print(f"\nRelaciones aprendidas por el modelo global (top {count} por Interest):")
    for interest in vocabulary:
        neighbors = nearest_words(interest, vocabulary, input_vectors, count=count)
        print(f"  {interest}")
        for related_interest, similarity in neighbors:
            print(f"    -> {related_interest} (coseno {similarity:.3f})")
            relations.append(
                {
                    "interest": interest,
                    "related_interest": related_interest,
                    "cosine_similarity": f"{similarity:.6f}",
                }
            )

    artifact_path = PROJECT_DIR / "server" / "artifacts" / "ndn_global_relations.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=("interest", "related_interest", "cosine_similarity"),
        )
        writer.writeheader()
        writer.writerows(relations)
    print(f"Relaciones completas: {artifact_path.relative_to(PROJECT_DIR)}")


def print_comparison(
    vocabulary: list[str], input_vectors: np.ndarray, output_vectors: np.ndarray
) -> None:
    probes = make_comparison_probes()
    summaries, report_rows = compare_forwarding_policies(
        probes, vocabulary, input_vectors, output_vectors
    )
    print(f"\nComparativa de {len(probes)} Interests (umbral coseno {SIMILARITY_THRESHOLD:.0%}):")
    print(
        f"{'Politica':<12} {'Data OK':>9} {'FP':>5} "
        f"{'Validos perdidos':>17} {'Bloqueados':>11} {'Sin Data':>9}"
    )
    for policy, label in (("baseline", "Sin modelo"), ("word2vec", "Word2Vec")):
        metrics = summaries[policy]
        print(
            f"{label:<12} {metrics['correct_data']:>9}"
            f" {metrics['false_positives']:>5}"
            f" {metrics['valid_missed']:>17}"
            f" {metrics['safely_blocked']:>11}"
            f" {metrics['no_data']:>9}"
        )
        if policy == "word2vec":
            print(
                f"  Detalle Word2Vec: {metrics['wrong_destination']} destinos erróneos, "
                f"{metrics['below_threshold']} rechazados por umbral/contexto, "
                f"{metrics['no_data']} Interests sin Data."
            )

    false_positive_rows = [
        row for row in report_rows if row["word2vec_outcome"] == "false_positive"
    ]
    print("Falsos positivos Word2Vec:")
    if false_positive_rows:
        for row in false_positive_rows[:5]:
            print(
                f"  {row['interest_received']} -> {row['word2vec_forwarded']} "
                f"(coseno {row['cosine_similarity']})"
            )
    else:
        print("  ninguno")

    artifact_path = PROJECT_DIR / "server" / "artifacts" / "ndn_100_interest_comparison.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Detalle por Interest: {artifact_path.relative_to(PROJECT_DIR)}")


def run(rounds: int = 20, local_epochs: int = 2, learning_rate: float = 0.1) -> None:
    if rounds < 1 or local_epochs < 1 or learning_rate <= 0:
        raise ValueError("rounds, local_epochs y learning_rate deben ser positivos")

    network = build_network()
    local_sessions, examples = collect_local_sessions(network)
    vocabulary = sorted({name for sessions in local_sessions.values() for session in sessions for name in session})
    local_word_ids = _to_word_ids(local_sessions, vocabulary)
    input_vectors, output_vectors = initialize_embeddings(
        len(vocabulary), EMBEDDING_DIM
    )

    print("FIB agregada (prefijo -> siguiente salto):")
    for node_name in ("edge_centro", "edge_norte", "edge_sur", "core"):
        forwarder = network.forwarders[node_name]
        for entry in forwarder.fib:
            print(f"  {node_name}: {entry.prefix} -> {entry.next_hop}")

    print("\nEjemplos de forwarding Interest -> Data:")
    for interest_name in (VIDEO_1, WEATHER_NORTE):
        result = examples[interest_name]
        matched_routes = ", ".join(
            f"{node}:{prefix}" for node, prefix, _ in result.fib_lookups
        )
        print(
            f"  {interest_name}\n"
            f"    Interest: {' -> '.join(result.interest_path)}\n"
            f"    Data:     {' -> '.join(result.data_path)}\n"
            f"    FIB:      {matched_routes}"
        )

    print(f"\nClientes FL: {len(local_word_ids)} | vocabulario Interest: {len(vocabulary)}")
    for round_number in range(1, rounds + 1):
        updates: list[tuple[np.ndarray, np.ndarray, int]] = []
        for client_index, sessions in enumerate(local_word_ids.values()):
            updated_input, updated_output, pair_count = train_local(
                sessions,
                input_vectors,
                output_vectors,
                epochs=local_epochs,
                learning_rate=learning_rate,
                seed=round_number * 100 + client_index,
            )
            updates.append((updated_input, updated_output, pair_count))

        total_pairs = sum(pair_count for _, _, pair_count in updates)
        input_vectors = sum(
            updated_input * pair_count
            for updated_input, _, pair_count in updates
        ) / total_pairs
        output_vectors = sum(
            updated_output * pair_count
            for _, updated_output, pair_count in updates
        ) / total_pairs
        print(
            f"Ronda {round_number:02d}/{rounds:02d} | "
            f"pares palabra-contexto agregados: {total_pairs}"
        )

    print_global_relations(vocabulary, input_vectors)
    print_comparison(vocabulary, input_vectors, output_vectors)

    artifact_path = PROJECT_DIR / "server" / "artifacts" / "ndn_global_model.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "algorithm": "federated_word2vec_skip_gram_negative_sampling",
                "rounds": rounds,
                "context_window": 2,
                "vocabulary": vocabulary,
                "input_vectors": input_vectors.tolist(),
                "output_vectors": output_vectors.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Modelo guardado en {artifact_path.relative_to(PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    try:
        run(args.rounds, args.local_epochs, args.learning_rate)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()