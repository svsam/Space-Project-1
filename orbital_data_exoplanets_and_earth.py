"""
Create a filtered Earth-like comparison CSV and graphs from orbital-quarter data.

Inputs created by generate_orbital_quarter_data.py:
    - exoplanet_orbital_quarter_data.csv
    - earth_orbital_quarter_data.csv

Output:
    - results.csv
    - planet_distance_vs_star_mass.png
    - exoplanet_vs_earth_midpoint_temperatures.png
    - orbit_speed_vs_orbit_period.png

This version compares each exoplanet quarter point to Earth at the same orbital
quarter using Earth's midpoint-eccentricity case only. A row is kept only if it
is sufficiently close to Earth's values. Rows that do not pass the filter are
excluded from results.csv and from every graph.

Default Earth-like filter:
    - orbital period within 50 percent of Earth
    - distance within 50 percent of Earth at the same quarter point
    - orbital speed within 50 percent of Earth at the same quarter point
    - host-star mass within 50 percent of the Sun
    - planet mass within a factor of 10 of Earth
    - global temperature within 50 K of Earth
    - north/equator/south temperatures within 50 K of Earth

Run:
    python compare_exoplanets_to_earth_quarters.py

Or:
    python compare_exoplanets_to_earth_quarters.py --exo exoplanet_orbital_quarter_data.csv --earth earth_orbital_quarter_data.csv --output results.csv

You can loosen or tighten the filter:
    python compare_exoplanets_to_earth_quarters.py --temp-tolerance-k 80 --period-relative-tolerance 1.0 --distance-relative-tolerance 1.0
"""

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
    "delta_orbital_period_years",
    "delta_distance_au",
    "delta_orbital_speed_m_s",
    "delta_host_mass_solar",
    "delta_planet_mass_earth",

    "similarity_score",
    "passed_earth_like_filter",
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


def relative_difference(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.inf

    if reference == 0:
        return math.inf

    return abs(value - reference) / abs(reference)


def absolute_difference(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.inf

    return abs(value - reference)


def get_earth_midpoint_rows(earth_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """
    Returns Earth rows for the midpoint eccentricity case, indexed by orbit quarter.
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


def earth_similarity_passes(
    *,
    exo_period: float,
    earth_period: float,
    exo_distance: float,
    earth_distance: float,
    exo_speed: float,
    earth_speed: float,
    exo_host_mass: float,
    earth_host_mass: float,
    exo_planet_mass: float,
    earth_planet_mass: float,
    exo_global: float,
    earth_global: float,
    exo_north: float,
    earth_north: float,
    exo_equator: float,
    earth_equator: float,
    exo_south: float,
    earth_south: float,
    period_relative_tolerance: float,
    distance_relative_tolerance: float,
    speed_relative_tolerance: float,
    host_mass_relative_tolerance: float,
    planet_mass_factor_tolerance: float,
    temp_tolerance_k: float,
) -> tuple[bool, float]:
    """
    Returns whether the exoplanet point is close enough to Earth and a similarity score.

    The score is smaller for more Earth-like points. A score near 0 means very
    close to Earth, while larger scores mean less similar.
    """
    period_rel = relative_difference(exo_period, earth_period)
    distance_rel = relative_difference(exo_distance, earth_distance)
    speed_rel = relative_difference(exo_speed, earth_speed)
    host_mass_rel = relative_difference(exo_host_mass, earth_host_mass)

    if exo_planet_mass > 0 and earth_planet_mass > 0:
        planet_mass_factor = max(exo_planet_mass / earth_planet_mass, earth_planet_mass / exo_planet_mass)
    else:
        planet_mass_factor = math.inf

    global_temp_abs = absolute_difference(exo_global, earth_global)
    north_temp_abs = absolute_difference(exo_north, earth_north)
    equator_temp_abs = absolute_difference(exo_equator, earth_equator)
    south_temp_abs = absolute_difference(exo_south, earth_south)

    passes = (
        period_rel <= period_relative_tolerance
        and distance_rel <= distance_relative_tolerance
        and speed_rel <= speed_relative_tolerance
        and host_mass_rel <= host_mass_relative_tolerance
        and planet_mass_factor <= planet_mass_factor_tolerance
        and global_temp_abs <= temp_tolerance_k
        and north_temp_abs <= temp_tolerance_k
        and equator_temp_abs <= temp_tolerance_k
        and south_temp_abs <= temp_tolerance_k
    )

    # Normalised score. Values below about 1 mean the row is within the chosen tolerances.
    score_parts = [
        period_rel / period_relative_tolerance if period_relative_tolerance > 0 else math.inf,
        distance_rel / distance_relative_tolerance if distance_relative_tolerance > 0 else math.inf,
        speed_rel / speed_relative_tolerance if speed_relative_tolerance > 0 else math.inf,
        host_mass_rel / host_mass_relative_tolerance if host_mass_relative_tolerance > 0 else math.inf,
        math.log10(planet_mass_factor) / math.log10(planet_mass_factor_tolerance)
        if planet_mass_factor > 0 and planet_mass_factor_tolerance > 1
        else math.inf,
        global_temp_abs / temp_tolerance_k if temp_tolerance_k > 0 else math.inf,
        north_temp_abs / temp_tolerance_k if temp_tolerance_k > 0 else math.inf,
        equator_temp_abs / temp_tolerance_k if temp_tolerance_k > 0 else math.inf,
        south_temp_abs / temp_tolerance_k if temp_tolerance_k > 0 else math.inf,
    ]

    finite_score_parts = [value for value in score_parts if math.isfinite(value)]
    similarity_score = sum(finite_score_parts) / len(finite_score_parts) if finite_score_parts else math.inf

    return passes, similarity_score


def build_results(
    exoplanet_rows: list[dict[str, str]],
    earth_midpoint_rows: dict[int, dict[str, str]],
    *,
    period_relative_tolerance: float,
    distance_relative_tolerance: float,
    speed_relative_tolerance: float,
    host_mass_relative_tolerance: float,
    planet_mass_factor_tolerance: float,
    temp_tolerance_k: float,
) -> tuple[list[dict[str, object]], int]:
    """
    Builds a compact filtered comparison table.

    Each exoplanet quarter is compared only with the Earth midpoint-eccentricity
    row at the same orbital quarter. The row is kept only if it passes the
    Earth-like similarity filter.
    """
    results: list[dict[str, object]] = []
    checked_rows = 0

    for exo in exoplanet_rows:
        quarter = int(parse_float(exo.get("orbit_quarter"), -1))

        if quarter not in earth_midpoint_rows:
            continue

        checked_rows += 1
        earth = earth_midpoint_rows[quarter]

        exo_period = parse_float(exo.get("orbital_period_years"))
        exo_distance = parse_float(exo.get("current_distance_au"))
        exo_speed = parse_float(exo.get("orbital_speed_m_s"))
        exo_host_mass = parse_float(exo.get("host_mass_solar"))
        exo_planet_mass = parse_float(exo.get("planet_mass_earth"))
        exo_e = parse_float(exo.get("eccentricity"))

        exo_global = parse_float(exo.get("global_temp_k"))
        exo_north = parse_float(exo.get("north_pole_temp_k"))
        exo_equator = parse_float(exo.get("equator_temp_k"))
        exo_south = parse_float(exo.get("south_pole_temp_k"))

        earth_period = parse_float(earth.get("orbital_period_years"))
        earth_distance = parse_float(earth.get("current_distance_au"))
        earth_speed = parse_float(earth.get("orbital_speed_m_s"))
        earth_host_mass = parse_float(earth.get("host_mass_solar"))
        earth_planet_mass = parse_float(earth.get("planet_mass_earth"))

        earth_global = parse_float(earth.get("global_temp_k"))
        earth_north = parse_float(earth.get("north_pole_temp_k"))
        earth_equator = parse_float(earth.get("equator_temp_k"))
        earth_south = parse_float(earth.get("south_pole_temp_k"))

        passes, similarity_score = earth_similarity_passes(
            exo_period=exo_period,
            earth_period=earth_period,
            exo_distance=exo_distance,
            earth_distance=earth_distance,
            exo_speed=exo_speed,
            earth_speed=earth_speed,
            exo_host_mass=exo_host_mass,
            earth_host_mass=earth_host_mass,
            exo_planet_mass=exo_planet_mass,
            earth_planet_mass=earth_planet_mass,
            exo_global=exo_global,
            earth_global=earth_global,
            exo_north=exo_north,
            earth_north=earth_north,
            exo_equator=exo_equator,
            earth_equator=earth_equator,
            exo_south=exo_south,
            earth_south=earth_south,
            period_relative_tolerance=period_relative_tolerance,
            distance_relative_tolerance=distance_relative_tolerance,
            speed_relative_tolerance=speed_relative_tolerance,
            host_mass_relative_tolerance=host_mass_relative_tolerance,
            planet_mass_factor_tolerance=planet_mass_factor_tolerance,
            temp_tolerance_k=temp_tolerance_k,
        )

        if not passes:
            continue

        results.append(
            {
                "planet": exo.get("planet", ""),
                "host_star": exo.get("host_star", ""),
                "orbit_quarter": quarter,
                "orbit_fraction": parse_float(exo.get("orbit_fraction")),

                "host_mass_solar": exo_host_mass,
                "planet_mass_earth": exo_planet_mass,
                "orbital_period_years": exo_period,
                "current_distance_au": exo_distance,
                "eccentricity": exo_e,
                "orbital_speed_m_s": exo_speed,

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
                "delta_orbital_period_years": exo_period - earth_period,
                "delta_distance_au": exo_distance - earth_distance,
                "delta_orbital_speed_m_s": exo_speed - earth_speed,
                "delta_host_mass_solar": exo_host_mass - earth_host_mass,
                "delta_planet_mass_earth": exo_planet_mass - earth_planet_mass,

                "similarity_score": similarity_score,
                "passed_earth_like_filter": True,
            }
        )

    return results, checked_rows


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

    if x_values and y_values:
        plt.scatter(x_values, y_values, s=14, alpha=0.7)
    else:
        plt.text(
            0.5,
            0.5,
            "No rows passed the Earth-like filter",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Host-star mass / solar masses")
    plt.ylabel("Planet-star distance / AU")
    plt.title("Earth-like Filtered Planet Distance vs Host-Star Mass")
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

    all_values: list[float] = []
    plotted_anything = False

    for label, earth_col, exo_col in locations:
        x_values: list[float] = []
        y_values: list[float] = []

        for row in rows:
            x = parse_float(row.get(earth_col))
            y = parse_float(row.get(exo_col))

            if finite_pair(x, y):
                x_values.append(x)
                y_values.append(y)
                all_values.extend([x, y])

        if x_values and y_values:
            plotted_anything = True
            plt.scatter(x_values, y_values, s=14, alpha=0.7, label=label)

    if all_values:
        low = min(all_values)
        high = max(all_values)
        plt.plot([low, high], [low, high], linestyle="--", linewidth=1, label="Equal temperature")

    if not plotted_anything:
        plt.text(
            0.5,
            0.5,
            "No rows passed the Earth-like filter",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        plt.legend()

    plt.xlabel("Earth midpoint-eccentricity temperature / K")
    plt.ylabel("Exoplanet temperature / K")
    plt.title("Filtered Exoplanet Regional Temperatures vs Earth")
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

    if x_values and y_values:
        plt.scatter(x_values, y_values, s=14, alpha=0.7)
        plt.xscale("log")
        plt.yscale("log")
    else:
        plt.text(
            0.5,
            0.5,
            "No positive period/speed values passed the Earth-like filter",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Orbital period / years")
    plt.ylabel("Orbital speed / m s$^{-1}$")
    plt.title("Earth-like Filtered Orbital Speed vs Orbital Period")
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
        description="Create Earth-like filtered exoplanet-Earth comparison CSV and graphs."
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
        help="Output filtered comparison CSV",
    )

    parser.add_argument(
        "--temp-tolerance-k",
        type=float,
        default=500.0,
        help="Maximum allowed temperature difference from Earth in K",
    )

    parser.add_argument(
        "--period-relative-tolerance",
        type=float,
        default=10,
        help="Maximum allowed fractional orbital-period difference from Earth",
    )

    parser.add_argument(
        "--distance-relative-tolerance",
        type=float,
        default=10,
        help="Maximum allowed fractional distance difference from Earth",
    )

    parser.add_argument(
        "--speed-relative-tolerance",
        type=float,
        default=10,
        help="Maximum allowed fractional orbital-speed difference from Earth",
    )

    parser.add_argument(
        "--host-mass-relative-tolerance",
        type=float,
        default=10,
        help="Maximum allowed fractional host-star mass difference from the Sun",
    )

    parser.add_argument(
        "--planet-mass-factor-tolerance",
        type=float,
        default=500.0,
        help="Maximum allowed mass factor difference from Earth mass",
    )

    args = parser.parse_args()

    exoplanet_rows = read_csv_rows(args.exo)
    earth_rows = read_csv_rows(args.earth)

    earth_midpoint_rows = get_earth_midpoint_rows(earth_rows)

    results, checked_rows = build_results(
        exoplanet_rows,
        earth_midpoint_rows,
        period_relative_tolerance=args.period_relative_tolerance,
        distance_relative_tolerance=args.distance_relative_tolerance,
        speed_relative_tolerance=args.speed_relative_tolerance,
        host_mass_relative_tolerance=args.host_mass_relative_tolerance,
        planet_mass_factor_tolerance=args.planet_mass_factor_tolerance,
        temp_tolerance_k=args.temp_tolerance_k,
    )

    write_results(args.output, results)
    make_graphs(results)

    print(f"Exoplanet quarter rows checked: {checked_rows}")
    print("Earth comparison case used: midpoint_eccentricity")
    print(f"Rows passing Earth-like filter: {len(results)}")
    print("")
    print(f"Rows removed from final CSV/graphs: {checked_rows - len(results)}")
    print(f"Saved filtered CSV: {args.output}")
    print(f"Saved graph: {DISTANCE_GRAPH}")
    print(f"Saved graph: {TEMPERATURE_GRAPH}")
    print(f"Saved graph: {SPEED_PERIOD_GRAPH}")

    if len(results) == 0:
        print("No rows passed the current filter. Try loosening the tolerances.")

if __name__ == "__main__":
    main()
