"""
Generate orbital-quarter temperature data for exoplanets and Earth.

This script is separate from the animation program. It reads the exoplanet CSV,
samples each exoplanet at four points in one orbit, and writes the results to a
new CSV file.

It also creates a separate Earth-only CSV using three eccentricity cases:
    1. minimum eccentricity limit
    2. midpoint eccentricity
    3. maximum eccentricity limit

For each planet and each quarter orbit, the script records:
    - orbital period
    - current distance
    - eccentricity
    - orbital speed
    - global equilibrium temperature
    - north pole temperature
    - equator temperature
    - south pole temperature

Run examples:
    python generate_orbital_quarter_data.py exoplanet_expected_eccentricities.csv
    python generate_orbital_quarter_data.py exoplanet_expected_eccentricities.csv --output exoplanet_quarters.csv --earth-output earth_quarters.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------

G = 6.67430e-11
AU_M = 1.495978707e11
M_SUN_KG = 1.98847e30
M_EARTH_KG = 5.9722e24
M_JUPITER_EARTH = 317.8
L_SUN_W = 3.828e26
SIGMA = 5.670374419e-8
DAYS_PER_YEAR = 365.25


# ------------------------------------------------------------
# Data model
# ------------------------------------------------------------

@dataclass
class PlanetSystem:
    row_index: int
    name: str
    host: str
    period_days: float
    semi_major_axis_au: float
    eccentricity: float
    planet_mass_earth: float
    host_mass_solar: float
    host_luminosity_solar: float
    albedo: float = 0.30
    axial_tilt_deg: float = 0.0

    @property
    def period_years(self) -> float:
        return self.period_days / DAYS_PER_YEAR

    @property
    def host_mass_kg(self) -> float:
        return self.host_mass_solar * M_SUN_KG

    @property
    def planet_mass_kg(self) -> float:
        return self.planet_mass_earth * M_EARTH_KG

    @property
    def luminosity_w(self) -> float:
        return self.host_luminosity_solar * L_SUN_W

    @property
    def mu(self) -> float:
        return G * (self.host_mass_kg + self.planet_mass_kg)


# ------------------------------------------------------------
# CSV parsing helpers
# ------------------------------------------------------------

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


def clamp_eccentricity(e: float) -> float:
    if not math.isfinite(e):
        return 0.0
    return min(max(e, 0.0), 0.95)


def first_valid_float(row: dict[str, str], names: list[str], default: float = math.nan) -> float:
    for name in names:
        value = parse_float(row.get(name), math.nan)
        if math.isfinite(value):
            return value
    return default


def host_luminosity_from_row(row: dict[str, str]) -> float:
    """
    Prefer st_lum_solar if it exists.
    Otherwise use NASA's st_lum, which is log10(L_star / L_sun).
    """
    lum_solar = parse_float(row.get("st_lum_solar"), math.nan)
    if math.isfinite(lum_solar) and lum_solar > 0:
        return lum_solar

    st_lum_log = parse_float(row.get("st_lum"), math.nan)
    if math.isfinite(st_lum_log):
        return 10.0 ** st_lum_log

    return 1.0


def eccentricity_from_row(row: dict[str, str]) -> float:
    eccentricity = first_valid_float(
        row,
        ["eccentricity_for_analysis", "expected_eccentricity", "pl_orbeccen"],
        default=0.0,
    )
    return clamp_eccentricity(eccentricity)


def planet_mass_from_row(row: dict[str, str]) -> float:
    mass_earth = first_valid_float(row, ["planet_mass_earth", "pl_bmasse"], math.nan)
    if math.isfinite(mass_earth) and mass_earth > 0:
        return mass_earth

    mass_jupiter = parse_float(row.get("pl_bmassj"), math.nan)
    if math.isfinite(mass_jupiter) and mass_jupiter > 0:
        return mass_jupiter * M_JUPITER_EARTH

    return 1.0


def period_from_a_if_missing(semi_major_axis_au: float, host_mass_solar: float) -> float:
    if semi_major_axis_au <= 0 or host_mass_solar <= 0:
        return math.nan
    period_years = math.sqrt(semi_major_axis_au ** 3 / host_mass_solar)
    return period_years * DAYS_PER_YEAR


def a_from_period_if_missing(period_days: float, host_mass_solar: float) -> float:
    if period_days <= 0 or host_mass_solar <= 0:
        return math.nan
    period_years = period_days / DAYS_PER_YEAR
    return (host_mass_solar * period_years ** 2) ** (1.0 / 3.0)


def row_to_planet(row: dict[str, str], row_index: int, forced_tilt_deg: float | None = None) -> PlanetSystem | None:
    name = (row.get("pl_name") or row.get("planet") or f"row_{row_index}").strip()
    host = (row.get("hostname") or row.get("host_star") or "Unknown host").strip()

    host_mass_solar = first_valid_float(row, ["st_mass", "host_mass_solar"], 1.0)
    if not math.isfinite(host_mass_solar) or host_mass_solar <= 0:
        host_mass_solar = 1.0

    period_days = first_valid_float(row, ["pl_orbper", "period_days"], math.nan)
    semi_major_axis_au = first_valid_float(row, ["pl_orbsmax", "semi_major_axis_au"], math.nan)

    if not math.isfinite(period_days) or period_days <= 0:
        period_days = period_from_a_if_missing(semi_major_axis_au, host_mass_solar)

    if not math.isfinite(semi_major_axis_au) or semi_major_axis_au <= 0:
        semi_major_axis_au = a_from_period_if_missing(period_days, host_mass_solar)

    if not math.isfinite(period_days) or period_days <= 0:
        return None
    if not math.isfinite(semi_major_axis_au) or semi_major_axis_au <= 0:
        return None

    axial_tilt_deg = first_valid_float(
        row,
        ["axial_tilt_deg", "obliquity_deg", "pl_obliq"],
        default=0.0,
    )
    if forced_tilt_deg is not None:
        axial_tilt_deg = forced_tilt_deg
    if not math.isfinite(axial_tilt_deg):
        axial_tilt_deg = 0.0

    albedo = first_valid_float(row, ["albedo", "planet_albedo"], 0.30)
    if not math.isfinite(albedo):
        albedo = 0.30
    albedo = min(max(albedo, 0.0), 0.95)

    return PlanetSystem(
        row_index=row_index,
        name=name,
        host=host,
        period_days=period_days,
        semi_major_axis_au=semi_major_axis_au,
        eccentricity=eccentricity_from_row(row),
        planet_mass_earth=planet_mass_from_row(row),
        host_mass_solar=host_mass_solar,
        host_luminosity_solar=host_luminosity_from_row(row),
        albedo=albedo,
        axial_tilt_deg=axial_tilt_deg,
    )


def load_exoplanets(csv_path: Path, forced_tilt_deg: float | None = None) -> list[PlanetSystem]:
    planets: list[PlanetSystem] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            planet = row_to_planet(row, row_index, forced_tilt_deg)
            if planet is not None:
                planets.append(planet)

    return planets


# ------------------------------------------------------------
# Orbital mechanics
# ------------------------------------------------------------

def solve_kepler(mean_anomaly: float, eccentricity: float, iterations: int = 12) -> float:
    E = mean_anomaly if eccentricity < 0.8 else math.pi

    for _ in range(iterations):
        f = E - eccentricity * math.sin(E) - mean_anomaly
        fp = 1.0 - eccentricity * math.cos(E)
        if abs(fp) < 1e-12:
            break
        E -= f / fp

    return E


def true_anomaly_from_fraction(orbit_fraction: float, eccentricity: float) -> float:
    mean_anomaly = 2.0 * math.pi * orbit_fraction
    eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)

    numerator = math.sqrt(1.0 + eccentricity) * math.sin(eccentric_anomaly / 2.0)
    denominator = math.sqrt(1.0 - eccentricity) * math.cos(eccentric_anomaly / 2.0)

    return 2.0 * math.atan2(numerator, denominator)


def orbital_radius_au(semi_major_axis_au: float, eccentricity: float, true_anomaly: float) -> float:
    return semi_major_axis_au * (1.0 - eccentricity ** 2) / (
        1.0 + eccentricity * math.cos(true_anomaly)
    )


def orbital_speed_m_s(planet: PlanetSystem, radius_au: float) -> float:
    radius_m = radius_au * AU_M
    semi_major_axis_m = planet.semi_major_axis_au * AU_M
    return math.sqrt(planet.mu * ((2.0 / radius_m) - (1.0 / semi_major_axis_m)))


# ------------------------------------------------------------
# Temperature model
# ------------------------------------------------------------

def global_equilibrium_temperature_k(planet: PlanetSystem, radius_au: float) -> float:
    radius_m = radius_au * AU_M
    return ((planet.luminosity_w * (1.0 - planet.albedo)) / (16.0 * math.pi * SIGMA * radius_m ** 2)) ** 0.25


def substellar_latitude_deg(planet: PlanetSystem, orbit_fraction: float) -> float:
    return planet.axial_tilt_deg * math.sin(2.0 * math.pi * orbit_fraction)


def local_temperature_k(global_temp_k: float, latitude_deg: float, substellar_lat_deg: float) -> float:
    """
    Simplified latitude-band temperature approximation.

    This keeps the same educational model as the animation file, but works in
    kelvin directly.
    """
    angular_distance = abs(latitude_deg - substellar_lat_deg)
    seasonal_heating = 16.0 * max(math.cos(math.radians(angular_distance)), 0.0)
    polar_cooling = 13.0 * (abs(latitude_deg) / 90.0) ** 1.15
    return global_temp_k + seasonal_heating - polar_cooling


def sample_planet_at_quarter(planet: PlanetSystem, orbit_fraction: float, case_label: str = "default") -> dict[str, object]:
    true_anomaly = true_anomaly_from_fraction(orbit_fraction, planet.eccentricity)
    radius_au = orbital_radius_au(
        planet.semi_major_axis_au,
        planet.eccentricity,
        true_anomaly,
    )
    speed_m_s = orbital_speed_m_s(planet, radius_au)

    global_temp_k = global_equilibrium_temperature_k(planet, radius_au)
    sub_lat = substellar_latitude_deg(planet, orbit_fraction)

    north_pole_temp_k = local_temperature_k(global_temp_k, 90.0, sub_lat)
    equator_temp_k = local_temperature_k(global_temp_k, 0.0, sub_lat)
    south_pole_temp_k = local_temperature_k(global_temp_k, -90.0, sub_lat)

    return {
        "case": case_label,
        "row_index": planet.row_index,
        "planet": planet.name,
        "host_star": planet.host,
        "orbit_fraction": orbit_fraction,
        "orbit_quarter": int(round(orbit_fraction * 4)),
        "orbital_period_years": planet.period_years,
        "semi_major_axis_au": planet.semi_major_axis_au,
        "current_distance_au": radius_au,
        "eccentricity": planet.eccentricity,
        "orbital_speed_m_s": speed_m_s,
        "global_temp_k": global_temp_k,
        "north_pole_temp_k": north_pole_temp_k,
        "equator_temp_k": equator_temp_k,
        "south_pole_temp_k": south_pole_temp_k,
    }


def sample_planet_orbit(planet: PlanetSystem, case_label: str = "default") -> list[dict[str, object]]:
    quarter_fractions = [0.0, 0.25, 0.50, 0.75]
    return [sample_planet_at_quarter(planet, fraction, case_label) for fraction in quarter_fractions]


# ------------------------------------------------------------
# Earth-only comparison cases
# ------------------------------------------------------------

def earth_case(eccentricity: float, case_label: str) -> PlanetSystem:
    return PlanetSystem(
        row_index=-1,
        name="Earth",
        host="Sun",
        period_days=365.25,
        semi_major_axis_au=1.0,
        eccentricity=eccentricity,
        planet_mass_earth=1.0,
        host_mass_solar=1.0,
        host_luminosity_solar=1.0,
        albedo=0.30,
        axial_tilt_deg=23.44,
    )


def generate_earth_rows() -> list[dict[str, object]]:
    """
    Creates Earth data for three eccentricity states.

    The minimum and maximum values are used as approximate eccentricity limits.
    The midpoint case is the centre between those limits.
    """
    e_min = 0.005
    e_max = 0.058
    e_mid = 0.5 * (e_min + e_max)

    cases = [
        earth_case(e_min, "minimum_eccentricity_limit"),
        earth_case(e_mid, "midpoint_eccentricity"),
        earth_case(e_max, "maximum_eccentricity_limit"),
    ]

    rows: list[dict[str, object]] = []
    for planet in cases:
        rows.extend(sample_planet_orbit(planet, case_label=planet.name + "_" + str(planet.eccentricity)))

    # Replace the less-readable auto labels with clearer labels.
    labels = [
        "minimum_eccentricity_limit",
        "midpoint_eccentricity",
        "maximum_eccentricity_limit",
    ]
    labelled_rows: list[dict[str, object]] = []
    for label, planet in zip(labels, cases):
        for row in sample_planet_orbit(planet, case_label=label):
            labelled_rows.append(row)

    return labelled_rows


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_COLUMNS = [
    "case",
    "row_index",
    "planet",
    "host_star",
    "orbit_fraction",
    "orbit_quarter",
    "orbital_period_years",
    "semi_major_axis_au",
    "current_distance_au",
    "eccentricity",
    "orbital_speed_m_s",
    "global_temp_k",
    "north_pole_temp_k",
    "equator_temp_k",
    "south_pole_temp_k",
]


def write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate orbital-quarter temperature CSV files.")
    parser.add_argument("csv_file", type=Path, help="Input exoplanet CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exoplanet_orbital_quarter_data.csv"),
        help="Output CSV for all exoplanets",
    )
    parser.add_argument(
        "--earth-output",
        type=Path,
        default=Path("earth_orbital_quarter_data.csv"),
        help="Output CSV for Earth comparison data",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=None,
        help="Optional axial tilt override for all exoplanets, in degrees",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(f"Could not find input CSV: {args.csv_file}")

    exoplanets = load_exoplanets(args.csv_file, forced_tilt_deg=args.tilt)
    if not exoplanets:
        raise ValueError("No usable exoplanets were loaded from the CSV.")

    exoplanet_rows: list[dict[str, object]] = []
    for planet in exoplanets:
        exoplanet_rows.extend(sample_planet_orbit(planet, case_label="exoplanet"))

    earth_rows = generate_earth_rows()

    write_rows(args.output, exoplanet_rows)
    write_rows(args.earth_output, earth_rows)

    print(f"Loaded exoplanets: {len(exoplanets)}")
    print(f"Exoplanet rows written: {len(exoplanet_rows)}")
    print(f"Earth rows written: {len(earth_rows)}")
    print(f"Saved exoplanet CSV: {args.output}")
    print(f"Saved Earth CSV: {args.earth_output}")


if __name__ == "__main__":
    main()
