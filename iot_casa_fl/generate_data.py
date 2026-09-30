"""Genera telemetria horaria sintetica y no-IID para tres hogares."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CLIENTS = {
    "casa_centro": {
        "description": "Piso urbano con teletrabajo entre semana",
        "profile": "remote_worker",
        "indoor_temp_c": 21.5,
        "outdoor_offset_c": 1.0,
        "appliance_intensity": 0.75,
        "seed": 1201,
    },
    "casa_norte": {
        "description": "Vivienda familiar en zona fresca",
        "profile": "family",
        "indoor_temp_c": 20.0,
        "outdoor_offset_c": -3.0,
        "appliance_intensity": 1.0,
        "seed": 2302,
    },
    "casa_sur": {
        "description": "Vivienda con ocupacion diurna frecuente y clima calido",
        "profile": "daytime",
        "indoor_temp_c": 23.0,
        "outdoor_offset_c": 5.0,
        "appliance_intensity": 0.9,
        "seed": 3403,
    },
}

def occupancy_probability(profile: str, hour: int, weekend: bool) -> float:
    if profile == "remote_worker":
        if 8 <= hour < 17:
            return 0.93
        if 18 <= hour < 23:
            return 0.76
        return 0.12
    if profile == "family":
        if weekend and 9 <= hour < 22:
            return 0.88
        if 6 <= hour < 8 or 17 <= hour < 23:
            return 0.87
        return 0.18
    if 9 <= hour < 21:
        return 0.94
    return 0.32


def make_rows(config: dict[str, object]) -> list[dict[str, object]]:
    rng = random.Random(int(config["seed"]))
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 6)

    for offset in range(30 * 24):
        timestamp = start + timedelta(hours=offset)
        hour = timestamp.hour
        weekend = timestamp.weekday() >= 5
        outdoor_temp = (
            10.0
            + float(config["outdoor_offset_c"])
            + 7.0 * math.sin(2.0 * math.pi * (timestamp.timetuple().tm_yday - 30) / 365.0)
            + 4.5 * math.sin(2.0 * math.pi * (hour - 8) / 24.0)
            + rng.gauss(0.0, 1.8)
        )
        occupied = int(rng.random() < occupancy_probability(str(config["profile"]), hour, weekend))
        indoor_temp = (
            float(config["indoor_temp_c"])
            + (outdoor_temp - float(config["indoor_temp_c"])) * 0.1
            + rng.gauss(0.0, 0.8)
        )
        humidity = max(25.0, min(85.0, 55.0 - (indoor_temp - 20.0) * 0.7 + rng.gauss(0, 5)))
        daylight = max(0.0, math.sin(math.pi * (hour - 6) / 12.0))
        lighting = 25.0 + daylight * 120.0
        if occupied and (hour < 7 or hour >= 19):
            lighting += rng.uniform(180.0, 650.0)
        appliance_w = 65.0 + rng.uniform(300.0, 1500.0) * float(config["appliance_intensity"]) * int(
            occupied and rng.random() < 0.42
        )
        hvac_on = int((outdoor_temp < 7.0 and indoor_temp < 21.0) or (outdoor_temp > 29.0 and indoor_temp > 24.0))
        hvac_energy = hvac_on * rng.uniform(0.7, 1.5)
        energy = max(
            0.1,
            0.2
            + appliance_w / 1000.0
            + hvac_energy
            + occupied * rng.uniform(0.08, 0.22)
            + rng.gauss(0.0, 0.07),
        )
        rows.append(
            {
                "timestamp": timestamp.isoformat(timespec="minutes"),
                "hour": hour,
                "indoor_temp_c": round(indoor_temp, 2),
                "outdoor_temp_c": round(outdoor_temp, 2),
                "humidity_pct": round(humidity, 2),
                "occupancy": occupied,
                "light_lux": round(lighting, 1),
                "appliance_w": round(appliance_w, 1),
                "hvac_on": hvac_on,
                "energy_kwh": round(energy, 3),
            }
        )
    return rows


def row_to_words(row: dict[str, object]) -> list[str]:
    hour = int(row["hour"])
    period = "noche" if hour < 6 or hour >= 22 else "manana" if hour < 12 else "tarde" if hour < 18 else "atardecer"
    indoor = float(row["indoor_temp_c"])
    outdoor = float(row["outdoor_temp_c"])
    humidity = float(row["humidity_pct"])
    words = [
        f"hora_{period}",
        "hogar_ocupado" if row["occupancy"] else "hogar_vacio",
        "consumo_alto" if float(row["energy_kwh"]) >= 1.2 else "consumo_medio" if float(row["energy_kwh"]) >= 0.7 else "consumo_bajo",
        "aparatos_activos" if float(row["appliance_w"]) >= 400 else "aparatos_base",
        "clima_encendido" if row["hvac_on"] else "clima_apagado",
        "interior_calido" if indoor >= 24 else "interior_fresco" if indoor < 19 else "interior_templado",
        "exterior_caluroso" if outdoor >= 25 else "exterior_frio" if outdoor < 8 else "exterior_templado",
        "humedad_alta" if humidity >= 65 else "humedad_baja" if humidity < 40 else "humedad_media",
        "luces_encendidas" if float(row["light_lux"]) >= 200 else "luces_apagadas",
    ]
    return words


def make_daily_sentences(config: dict[str, object]) -> list[str]:
    rows = make_rows(config)
    sentences: list[str] = []
    for day_start in range(0, len(rows), 24):
        daily_words = [
            word
            for row in rows[day_start : day_start + 24]
            for word in row_to_words(row)
        ]
        sentences.append(" ".join(daily_words))
    return sentences


def write_corpus(path: Path, sentences: list[str]) -> None:
    path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def main() -> None:
    all_sentences: list[str] = []
    for client_id, config in CLIENTS.items():
        client_dir = PROJECT_DIR / "clients" / client_id
        data_dir = client_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (client_dir / "config.json").write_text(
            json.dumps({"client_id": client_id, **config}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        sentences = make_daily_sentences(config)
        split = int(len(sentences) * 0.8)
        all_sentences.extend(sentences[:split])
        for old_csv in (data_dir / "train.csv", data_dir / "test.csv"):
            old_csv.unlink(missing_ok=True)
        write_corpus(data_dir / "train.txt", sentences[:split])
        write_corpus(data_dir / "test.txt", sentences[split:])
        print(f"{client_id}: {split} dias train, {len(sentences) - split} dias test")

    vocabulary = sorted(
        {word for sentence in all_sentences for word in sentence.split()}
    )
    vocabulary_path = PROJECT_DIR / "shared" / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Vocabulario compartido: {len(vocabulary)} palabras")


if __name__ == "__main__":
    main()