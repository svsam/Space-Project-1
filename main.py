from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


EXOPLANET_DEFAULT = Path("exoplanet_orbital_quarter_data.csv")
EARTH_DEFAULT = Path("earth_orbital_quarter_data.csv")
OUTPUT_DEFAULT = Path("results.csv")

DISTANCE_GRAPH = Path("planet_distance_vs_star_mass.png")
TEMPERATURE_GRAPH = Path("exoplanet_vs_earth_midpoint_temperatures.png")
SPEED_PERIOD_GRAPH = Path("orbit_speed_vs_orbit_period.png")


RESULT_COLUMNS = [
    "planet",
    "host_star",
    "orbit_quarter",
    "orbit_fraction",

    "host_mass_solar",
    "planet_mass_earth",
    "orbital_period_years",
    "current_distance_au",
    "eccentricity",
    "orbital_speed_m_s",

    "exoplanet_global_temp_k",
    "exoplanet_north_pole_temp_k",
    "exoplanet_equator_temp_k",
    "exoplanet_south_pole_temp_k",

    "earth_case",
    "earth_global_temp_k",
    "earth_north_pole_temp_k",
    "earth_equator_temp_k",
    "earth_south_pole_temp_k",

    "delta_global_temp_k",
    "delta_north_pole_temp_k",
    "delta_equator_temp_k",
    "delta_south_pole_temp_k",
]


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "null", "--"}:
        return default

    try:
        return float(text)
    except ValueError:
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def finite_pair(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)


def get_earth_midpoint_rows(earth_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """
    Returns Earth rows for the midpoint eccentricity case, indexed by orbit quarter.

    The Earth CSV contains three cases:
        - minimum_eccentricity_limit
        - midpoint_eccentricity
        - maximum_eccentricity_limit

    Only midpoint_eccentricity is used for the final comparison and temperature graph.
    """
    midpoint_rows: dict[int, dict[str, str]] = {}

    for row in earth_rows:
        if row.get("case") != "midpoint_eccentricity":
            continue

        quarter = int(parse_float(row.get("orbit_quarter"), -1))

        if quarter >= 0:
            midpoint_rows[quarter] = row

    if len(midpoint_rows) != 4:
        raise ValueError(
            "Could not find all four Earth midpoint-eccentricity quarter rows. "
            "Regenerate earth_orbital_quarter_data.csv first."
        )

    return midpoint_rows


def build_results(
    exoplanet_rows: list[dict[str, str]],
    earth_midpoint_rows: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    """
    Builds a compact comparison table.

    Each exoplanet quarter is compared only with the Earth midpoint-eccentricity
    row at the same orbital quarter. This keeps one result row per exoplanet
    quarter point.
    """
    results: list[dict[str, object]] = []

    for exo in exoplanet_rows:
        quarter = int(parse_float(exo.get("orbit_quarter"), -1))

        if quarter not in earth_midpoint_rows:
            continue

        earth = earth_midpoint_rows[quarter]

        exo_global = parse_float(exo.get("global_temp_k"))
        exo_north = parse_float(exo.get("north_pole_temp_k"))
        exo_equator = parse_float(exo.get("equator_temp_k"))
        exo_south = parse_float(exo.get("south_pole_temp_k"))

        earth_global = parse_float(earth.get("global_temp_k"))
        earth_north = parse_float(earth.get("north_pole_temp_k"))
        earth_equator = parse_float(earth.get("equator_temp_k"))
        earth_south = parse_float(earth.get("south_pole_temp_k"))

        results.append(
            {
                "planet": exo.get("planet", ""),
                "host_star": exo.get("host_star", ""),
                "orbit_quarter": quarter,
                "orbit_fraction": parse_float(exo.get("orbit_fraction")),

                "host_mass_solar": parse_float(exo.get("host_mass_solar")),
                "planet_mass_earth": parse_float(exo.get("planet_mass_earth")),
                "orbital_period_years": parse_float(exo.get("orbital_period_years")),
                "current_distance_au": parse_float(exo.get("current_distance_au")),
                "eccentricity": parse_float(exo.get("eccentricity")),
                "orbital_speed_m_s": parse_float(exo.get("orbital_speed_m_s")),

                "exoplanet_global_temp_k": exo_global,
                "exoplanet_north_pole_temp_k": exo_north,
                "exoplanet_equator_temp_k": exo_equator,
                "exoplanet_south_pole_temp_k": exo_south,

                "earth_case": earth.get("case", ""),
                "earth_global_temp_k": earth_global,
                "earth_north_pole_temp_k": earth_north,
                "earth_equator_temp_k": earth_equator,
                "earth_south_pole_temp_k": earth_south,

                "delta_global_temp_k": exo_global - earth_global,
                "delta_north_pole_temp_k": exo_north - earth_north,
                "delta_equator_temp_k": exo_equator - earth_equator,
                "delta_south_pole_temp_k": exo_south - earth_south,
            }
        )

    return results


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_distance_vs_star_mass_graph(rows: list[dict[str, object]], output_path: Path) -> None:
    x_values: list[float] = []
    y_values: list[float] = []

    for row in rows:
        x = parse_float(row.get("host_mass_solar"))
        y = parse_float(row.get("current_distance_au"))

        if finite_pair(x, y):
            x_values.append(x)
            y_values.append(y)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, s=12, alpha=0.7)
    plt.xlabel("Host-star mass / solar masses")
    plt.ylabel("Planet-star distance / AU")
    plt.title("Planet Distance vs Host-Star Mass")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_temperature_comparison_graph(rows: list[dict[str, object]], output_path: Path) -> None:
    locations = [
        (
            "North pole",
            "earth_north_pole_temp_k",
            "exoplanet_north_pole_temp_k",
        ),
        (
            "Equator",
            "earth_equator_temp_k",
            "exoplanet_equator_temp_k",
        ),
        (
            "South pole",
            "earth_south_pole_temp_k",
            "exoplanet_south_pole_temp_k",
        ),
    ]

    plt.figure(figsize=(8, 5))

    for label, earth_col, exo_col in locations:
        x_values: list[float] = []
        y_values: list[float] = []

        for row in rows:
            x = parse_float(row.get(earth_col))
            y = parse_float(row.get(exo_col))

            if finite_pair(x, y):
                x_values.append(x)
                y_values.append(y)

        plt.scatter(x_values, y_values, s=12, alpha=0.7, label=label)

    plt.xlabel("Earth midpoint-eccentricity temperature / K")
    plt.ylabel("Exoplanet temperature / K")
    plt.title("Exoplanet Regional Temperatures vs Earth Midpoint-Eccentricity Temperatures")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_speed_vs_period_graph(rows: list[dict[str, object]], output_path: Path) -> None:
    x_values: list[float] = []
    y_values: list[float] = []

    for row in rows:
        x = parse_float(row.get("orbital_period_years"))
        y = parse_float(row.get("orbital_speed_m_s"))

        if finite_pair(x, y) and x > 0 and y > 0:
            x_values.append(x)
            y_values.append(y)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, s=12, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Orbital period / years")
    plt.ylabel("Orbital speed / m s$^{-1}$")
    plt.title("Orbital Speed vs Orbital Period")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_graphs(rows: list[dict[str, object]]) -> None:
    make_distance_vs_star_mass_graph(rows, DISTANCE_GRAPH)
    make_temperature_comparison_graph(rows, TEMPERATURE_GRAPH)
    make_speed_vs_period_graph(rows, SPEED_PERIOD_GRAPH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create slim exoplanet-Earth comparison CSV and graphs."
    )

    parser.add_argument(
        "--exo",
        type=Path,
        default=EXOPLANET_DEFAULT,
        help="Input exoplanet quarter-data CSV",
    )

    parser.add_argument(
        "--earth",
        type=Path,
        default=EARTH_DEFAULT,
        help="Input Earth quarter-data CSV",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Output comparison CSV",
    )

    args = parser.parse_args()

    exoplanet_rows = read_csv_rows(args.exo)
    earth_rows = read_csv_rows(args.earth)

    earth_midpoint_rows = get_earth_midpoint_rows(earth_rows)
    results = build_results(exoplanet_rows, earth_midpoint_rows)

    write_results(args.output, results)
    make_graphs(results)

    print(f"Exoplanet quarter rows read: {len(exoplanet_rows)}")
    print("Earth comparison case used: midpoint_eccentricity")
    print(f"Slim comparison rows written: {len(results)}")
    print(f"Saved CSV: {args.output}")
    print(f"Saved graph: {DISTANCE_GRAPH}")
    print(f"Saved graph: {TEMPERATURE_GRAPH}")
    print(f"Saved graph: {SPEED_PERIOD_GRAPH}")


if __name__ == "__main__":
    main()
