"""
Exoplanet orbit and temperature animation.

This program reads an optional CSV of exoplanet data, lets you choose a planet
from a terminal menu using numbers from 1 to n, then animates the orbit and
records idealised temperatures.

If no CSV file is supplied, or if you choose the built-in fallback, the program
uses Earth orbiting the Sun.

Run examples:
    python exoplanet_orbit_temperature_sim.py
    python exoplanet_orbit_temperature_sim.py exoplanet_expected_eccentricities.csv
    python exoplanet_orbit_temperature_sim.py exoplanet_expected_eccentricities.csv --choice 12
    python exoplanet_orbit_temperature_sim.py exoplanet_expected_eccentricities.csv --choice 12 --single
    python exoplanet_orbit_temperature_sim.py exoplanet_expected_eccentricities.csv --years 10000 --duration 60

Expected useful CSV columns:
    pl_name, hostname, pl_orbper, pl_orbsmax, st_mass, st_lum or st_lum_solar,
    pl_bmasse, pl_bmassj, eccentricity_for_analysis, expected_eccentricity,
    pl_orbeccen
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
# Display constants
# ------------------------------------------------------------

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 780
SIDEBAR_WIDTH = 370
FPS_MS = 33
ORBIT_POINTS = 360
LATITUDE_BANDS = [-90, -60, -30, 0, 30, 60, 90]


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Exoplanet:
    row_index: int
    menu_index: int
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
# Built-in Earth/Sun fallback
# ------------------------------------------------------------

def earth_sun_planet() -> Exoplanet:
    return Exoplanet(
        row_index=-1,
        menu_index=1,
        name="Earth",
        host="Sun",
        period_days=365.25,
        semi_major_axis_au=1.0,
        eccentricity=0.0167,
        planet_mass_earth=1.0,
        host_mass_solar=1.0,
        host_luminosity_solar=1.0,
        albedo=0.30,
        axial_tilt_deg=23.44,
    )


# ------------------------------------------------------------
# CSV loading and cleaning
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


def first_valid_float(row: dict[str, str], names: list[str], default: float = math.nan) -> float:
    for name in names:
        value = parse_float(row.get(name), math.nan)
        if math.isfinite(value):
            return value
    return default


def host_luminosity_from_row(row: dict[str, str]) -> float:
    """
    Prefer st_lum_solar if your earlier script created it.
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
    """
    Prefer the column made by the statistical eccentricity program.
    Then fall back to the expected value, then the raw NASA value.
    """
    eccentricity = first_valid_float(
        row,
        ["eccentricity_for_analysis", "expected_eccentricity", "pl_orbeccen"],
        default=0.0,
    )

    if not math.isfinite(eccentricity):
        return 0.0

    return min(max(eccentricity, 0.0), 0.95)


def planet_mass_from_row(row: dict[str, str]) -> float:
    mass_earth = first_valid_float(row, ["planet_mass_earth", "pl_bmasse"], math.nan)
    if math.isfinite(mass_earth) and mass_earth > 0:
        return mass_earth

    mass_jupiter = parse_float(row.get("pl_bmassj"), math.nan)
    if math.isfinite(mass_jupiter) and mass_jupiter > 0:
        return mass_jupiter * M_JUPITER_EARTH

    return 1.0


def period_from_a_if_missing(semi_major_axis_au: float, host_mass_solar: float) -> float:
    """
    Kepler's third law in Solar-system units:
        P^2 = a^3 / M_star
    where P is in years, a is in AU, and M_star is in solar masses.
    """
    if semi_major_axis_au <= 0 or host_mass_solar <= 0:
        return math.nan
    period_years = math.sqrt(semi_major_axis_au ** 3 / host_mass_solar)
    return period_years * DAYS_PER_YEAR


def a_from_period_if_missing(period_days: float, host_mass_solar: float) -> float:
    """
    Rearranged Kepler's third law:
        a = (M_star P^2)^(1/3)
    where P is in years and a is in AU.
    """
    if period_days <= 0 or host_mass_solar <= 0:
        return math.nan
    period_years = period_days / DAYS_PER_YEAR
    return (host_mass_solar * period_years ** 2) ** (1.0 / 3.0)


def row_to_exoplanet(
    row: dict[str, str],
    row_index: int,
    menu_index: int,
    forced_tilt_deg: float | None,
) -> Exoplanet | None:
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

    return Exoplanet(
        row_index=row_index,
        menu_index=menu_index,
        name=name,
        host=host,
        period_days=period_days,
        semi_major_axis_au=semi_major_axis_au,
        eccentricity=eccentricity_from_row(row),
        planet_mass_earth=planet_mass_from_row(row),
        host_mass_solar=host_mass_solar,
        host_luminosity_solar=host_luminosity_from_row(row),
        albedo=first_valid_float(row, ["albedo", "planet_albedo"], 0.30),
        axial_tilt_deg=axial_tilt_deg,
    )


def load_planets(csv_path: Path, forced_tilt_deg: float | None) -> list[Exoplanet]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        planets = []
        menu_index = 1
        for row_index, row in enumerate(reader):
            planet = row_to_exoplanet(row, row_index, menu_index, forced_tilt_deg)
            if planet is not None:
                planets.append(planet)
                menu_index += 1
    return planets


def print_planet_menu(planets: list[Exoplanet], limit: int = 80) -> None:
    print("\nChoose a planet:")
    print("number | planet | host | P / days | a / AU | e")
    print("-" * 92)
    for planet in planets[:limit]:
        print(
            f"{planet.menu_index:6d} | "
            f"{planet.name[:24]:24s} | "
            f"{planet.host[:20]:20s} | "
            f"{planet.period_days:9.3f} | "
            f"{planet.semi_major_axis_au:7.4f} | "
            f"{planet.eccentricity:5.3f}"
        )
    if len(planets) > limit:
        print(f"... {len(planets) - limit} more not shown. Use --choice N to choose one directly.")
    print()


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


def true_anomaly_from_time(time_days: float, planet: Exoplanet) -> float:
    M = 2.0 * math.pi * ((time_days / planet.period_days) % 1.0)
    E = solve_kepler(M, planet.eccentricity)
    numerator = math.sqrt(1.0 + planet.eccentricity) * math.sin(E / 2.0)
    denominator = math.sqrt(1.0 - planet.eccentricity) * math.cos(E / 2.0)
    return 2.0 * math.atan2(numerator, denominator)


def relative_position_au(planet: Exoplanet, true_anomaly: float) -> tuple[float, float, float]:
    """
    Returns the planet position relative to a stationary host star at one focus.
    """
    e = planet.eccentricity
    a = planet.semi_major_axis_au
    r = a * (1.0 - e ** 2) / (1.0 + e * math.cos(true_anomaly))
    x = r * math.cos(true_anomaly)
    y = r * math.sin(true_anomaly)
    return x, y, r


def orbital_speed_m_s(planet: Exoplanet, radius_au: float) -> float:
    r_m = radius_au * AU_M
    a_m = planet.semi_major_axis_au * AU_M
    return math.sqrt(planet.mu * (2.0 / r_m - 1.0 / a_m))


# ------------------------------------------------------------
# Temperature model
# ------------------------------------------------------------

def global_equilibrium_temperature_k(planet: Exoplanet, radius_au: float) -> float:
    """
    Black-body equilibrium temperature with full surface redistribution:
        T = [ L(1-A)/(16 pi sigma r^2) ]^(1/4)
    """
    r_m = radius_au * AU_M
    return ((planet.luminosity_w * (1.0 - planet.albedo)) / (16.0 * math.pi * SIGMA * r_m ** 2)) ** 0.25


def substellar_latitude_deg(planet: Exoplanet, time_days: float) -> float:
    """
    Simple seasonal tilt model. If axial tilt is unavailable, this is zero.
    """
    orbital_phase = 2.0 * math.pi * ((time_days / planet.period_days) % 1.0)
    return planet.axial_tilt_deg * math.sin(orbital_phase)


def local_temperature_c(global_temp_c: float, latitude_deg: float, substellar_lat_deg: float) -> float:
    """
    Educational regional-temperature approximation.

    This is not a full atmospheric model. It shifts the warmest band toward the
    substellar latitude and cools polar regions.
    """
    angular_distance = abs(latitude_deg - substellar_lat_deg)
    seasonal_heating = 16.0 * max(math.cos(math.radians(angular_distance)), 0.0)
    polar_cooling = 13.0 * (abs(latitude_deg) / 90.0) ** 1.15
    return global_temp_c + seasonal_heating - polar_cooling


def temperature_to_color(temp_c: float, min_c: float, max_c: float) -> str:
    """
    Blue-white-orange-red gradient for planet bands.
    """
    anchors = [
        (min_c, (180, 220, 255)),
        ((min_c + max_c) * 0.38, (230, 245, 255)),
        ((min_c + max_c) * 0.55, (255, 220, 140)),
        ((min_c + max_c) * 0.75, (255, 145, 70)),
        (max_c, (215, 45, 45)),
    ]
    value = max(min_c, min(max_c, temp_c))
    for (ta, ca), (tb, cb) in zip(anchors, anchors[1:]):
        if ta <= value <= tb:
            factor = 0.0 if tb == ta else (value - ta) / (tb - ta)
            r = round(ca[0] + factor * (cb[0] - ca[0]))
            g = round(ca[1] + factor * (cb[1] - ca[1]))
            b = round(ca[2] + factor * (cb[2] - ca[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#d72d2d"


# ------------------------------------------------------------
# Tkinter animation engine
# ------------------------------------------------------------

class ExoplanetOrbitApp:
    def __init__(
        self,
        planets: list[Exoplanet],
        selected: Exoplanet,
        simulated_years: float,
        duration_seconds: float,
    ) -> None:
        self.planets = planets
        self.selected = selected
        self.simulated_years = simulated_years
        self.duration_seconds = duration_seconds
        self.start_time = time.perf_counter()
        self.paused = False
        self.pause_started_at = 0.0
        self.total_pause_time = 0.0
        self.finished = False

        self.root = tk.Tk()
        self.root.title("Exoplanet orbital-temperature simulation")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#08111f")

        self.canvas = tk.Canvas(
            self.root,
            width=WINDOW_WIDTH - SIDEBAR_WIDTH,
            height=WINDOW_HEIGHT,
            bg="#050a14",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.root, width=SIDEBAR_WIDTH, bg="#0d1424")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.metrics: dict[str, tk.Label] = {}
        self._build_sidebar()

        self.max_extent_au = max(
            p.semi_major_axis_au * (1.0 + p.eccentricity) for p in self.planets
        )
        self.max_extent_au = max(self.max_extent_au, 0.01)

        self.temp_min, self.temp_max = self._estimate_temperature_range()

        self.root.bind("<space>", lambda event: self.toggle_pause())
        self.root.bind("<Escape>", lambda event: self.root.destroy())

    def _build_sidebar(self) -> None:
        title = tk.Label(
            self.sidebar,
            text="Exoplanet Orbit Model",
            fg="white",
            bg="#0d1424",
            font=("Arial", 18, "bold"),
        )
        title.pack(anchor="w", padx=20, pady=(20, 8))

        description = (
            "The chosen planet is selected from the terminal menu. The host star "
            "is assumed to remain stationary at the centre of the system. The "
            f"animation compresses {self.simulated_years:,.0f} simulated years "
            f"into {self.duration_seconds:.0f} seconds."
        )
        tk.Label(
            self.sidebar,
            text=description,
            wraplength=320,
            justify="left",
            fg="#d9e8ff",
            bg="#0d1424",
        ).pack(anchor="w", padx=20, pady=(0, 12))

        for key in [
            "Selected planet",
            "Host star",
            "Simulated year",
            "Orbital period",
            "Semi-major axis",
            "Eccentricity",
            "Distance",
            "Orbital speed",
            "Global temp",
            "Substellar latitude",
            "North pole temp",
            "Equator temp",
            "South pole temp",
            "Animation status",
        ]:
            self._add_metric_row(key)

        legend_text = (
            "Space: pause/resume\n"
            "Esc: quit\n\n"
            "The star is fixed at the centre. This is a deliberate approximation "
            "because the host-star motion is usually much smaller than the planet's orbit."
        )
        tk.Label(
            self.sidebar,
            text=legend_text,
            wraplength=320,
            justify="left",
            fg="#9fb5d1",
            bg="#0d1424",
        ).pack(anchor="w", padx=20, pady=16)

    def _add_metric_row(self, key: str) -> None:
        frame = tk.Frame(self.sidebar, bg="#172138")
        frame.pack(fill=tk.X, padx=18, pady=5)
        tk.Label(
            frame,
            text=key,
            fg="white",
            bg="#172138",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(7, 0))
        value = tk.Label(
            frame,
            text="-",
            fg="#9fd3ff",
            bg="#172138",
            font=("Consolas", 10),
        )
        value.pack(anchor="w", padx=10, pady=(2, 7))
        self.metrics[key] = value

    def _estimate_temperature_range(self) -> tuple[float, float]:
        temps = []
        for planet in self.planets:
            peri = planet.semi_major_axis_au * (1.0 - planet.eccentricity)
            apo = planet.semi_major_axis_au * (1.0 + planet.eccentricity)
            if peri > 0:
                temps.append(global_equilibrium_temperature_k(planet, peri) - 273.15)
            if apo > 0:
                temps.append(global_equilibrium_temperature_k(planet, apo) - 273.15)
        if not temps:
            return -100.0, 100.0
        low = min(temps) - 30.0
        high = max(temps) + 30.0
        if abs(high - low) < 1.0:
            high = low + 1.0
        return low, high

    def toggle_pause(self) -> None:
        if self.finished:
            return
        if not self.paused:
            self.paused = True
            self.pause_started_at = time.perf_counter()
        else:
            self.paused = False
            self.total_pause_time += time.perf_counter() - self.pause_started_at
            self._tick()

    def elapsed_real_seconds(self) -> float:
        if self.paused:
            return self.pause_started_at - self.start_time - self.total_pause_time
        return time.perf_counter() - self.start_time - self.total_pause_time

    def simulated_time(self) -> tuple[float, float, float]:
        elapsed = min(max(self.elapsed_real_seconds(), 0.0), self.duration_seconds)
        fraction = elapsed / self.duration_seconds if self.duration_seconds > 0 else 1.0
        sim_year = fraction * self.simulated_years
        sim_days = sim_year * DAYS_PER_YEAR
        return elapsed, sim_year, sim_days

    def transform(self, x_au: float, y_au: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        scale = 0.43 * min(width, height) / self.max_extent_au
        return width * 0.5 + x_au * scale, height * 0.5 - y_au * scale

    def orbit_points(self, planet: Exoplanet) -> Iterable[tuple[float, float]]:
        for i in range(ORBIT_POINTS + 1):
            nu = 2.0 * math.pi * i / ORBIT_POINTS
            x, y, _ = relative_position_au(planet, nu)
            yield x, y

    def planet_state(self, planet: Exoplanet, sim_days: float) -> dict[str, float]:
        nu = true_anomaly_from_time(sim_days, planet)
        x, y, r = relative_position_au(planet, nu)
        speed = orbital_speed_m_s(planet, r)
        global_k = global_equilibrium_temperature_k(planet, r)
        global_c = global_k - 273.15
        sub_lat = substellar_latitude_deg(planet, sim_days)
        local = {
            lat: local_temperature_c(global_c, lat, sub_lat)
            for lat in LATITUDE_BANDS
        }
        return {
            "nu": nu,
            "x": x,
            "y": y,
            "r": r,
            "speed": speed,
            "global_c": global_c,
            "sub_lat": sub_lat,
            "north_pole_c": local[90],
            "equator_c": local[0],
            "south_pole_c": local[-90],
            **{f"lat_{lat}": temp for lat, temp in local.items()},
        }

    def draw_orbit(self, planet: Exoplanet, selected: bool) -> None:
        points: list[float] = []
        for x, y in self.orbit_points(planet):
            px, py = self.transform(x, y)
            points.extend([px, py])
        color = "#d6e7ff" if selected else "#4b617d"
        width = 3 if selected else 1
        dash = (7, 7) if selected else (4, 7)
        self.canvas.create_line(*points, fill=color, dash=dash, width=width, smooth=True)

    def draw_planet_disc(self, planet: Exoplanet, state: dict[str, float], px: float, py: float, selected: bool) -> None:
        radius = 15 if selected else 8
        band_step = max(1, int(radius / 8))
        sub_lat = state["sub_lat"]
        global_c = state["global_c"]

        for pixel_y in range(int(py - radius), int(py + radius) + 1, band_step):
            dy = pixel_y - py
            if abs(dy) > radius:
                continue
            chord = math.sqrt(max(radius ** 2 - dy ** 2, 0.0))
            latitude = 90.0 * (dy / radius)
            temp = local_temperature_c(global_c, latitude, sub_lat)
            color = temperature_to_color(temp, self.temp_min, self.temp_max)
            self.canvas.create_line(px - chord, pixel_y, px + chord, pixel_y, fill=color, width=band_step)

        outline = "#ffffff" if selected else "#c8d8ed"
        self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, outline=outline, width=2)

        if selected:
            axis_angle = math.radians(90.0 - planet.axial_tilt_deg)
            dx = math.cos(axis_angle) * radius * 0.95
            dy = -math.sin(axis_angle) * radius * 0.95
            self.canvas.create_line(px - dx, py - dy, px + dx, py + dy, fill="#e8f3ff", width=2)
            self.canvas.create_text(px, py + radius + 13, text=planet.name, fill="#e8f3ff", font=("Arial", 9, "bold"))

    def draw_frame(self) -> None:
        self.canvas.delete("all")
        elapsed, sim_year, sim_days = self.simulated_time()

        states = {planet.name: self.planet_state(planet, sim_days) for planet in self.planets}

        for planet in self.planets:
            self.draw_orbit(planet, planet.menu_index == self.selected.menu_index)

        sx, sy = self.transform(0.0, 0.0)
        self.canvas.create_oval(sx - 18, sy - 18, sx + 18, sy + 18, fill="#ffca45", outline="")
        self.canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7, fill="#fff2a6", outline="")
        self.canvas.create_text(sx, sy + 31, text=self.selected.host, fill="#ffe9a8", font=("Arial", 10, "bold"))

        for planet in self.planets:
            state = states[planet.name]
            selected = planet.menu_index == self.selected.menu_index
            px, py = self.transform(state["x"], state["y"])
            self.draw_planet_disc(planet, state, px, py, selected)

        selected_state = states[self.selected.name]
        self.update_sidebar(elapsed, sim_year, selected_state)

        self.canvas.create_text(
            22,
            24,
            text="Exoplanet system orbit and temperature model",
            fill="#d6e7ff",
            font=("Arial", 13, "bold"),
            anchor="w",
        )
        self.canvas.create_text(
            22,
            45,
            text="Dashed curves: Keplerian orbits. Star fixed at one focus. Planet colours: idealised latitude-band temperatures.",
            fill="#9fb5d1",
            font=("Arial", 10),
            anchor="w",
        )

    def update_sidebar(self, elapsed: float, sim_year: float, state: dict[str, float]) -> None:
        self.metrics["Selected planet"].config(text=f"{self.selected.name}  [choice {self.selected.menu_index}]")
        self.metrics["Host star"].config(text=self.selected.host)
        self.metrics["Simulated year"].config(text=f"{sim_year:,.1f} / {self.simulated_years:,.0f}")
        self.metrics["Orbital period"].config(text=f"{self.selected.period_days:,.4f} days")
        self.metrics["Semi-major axis"].config(text=f"{self.selected.semi_major_axis_au:.5f} AU")
        self.metrics["Eccentricity"].config(text=f"{self.selected.eccentricity:.5f}")
        self.metrics["Distance"].config(text=f"{state['r']:.5f} AU")
        self.metrics["Orbital speed"].config(text=f"{state['speed'] / 1000.0:.3f} km/s")
        self.metrics["Global temp"].config(text=f"{state['global_c']:.2f} °C")
        self.metrics["Substellar latitude"].config(text=f"{state['sub_lat']:.2f}°")
        self.metrics["North pole temp"].config(text=f"{state['north_pole_c']:.2f} °C")
        self.metrics["Equator temp"].config(text=f"{state['equator_c']:.2f} °C")
        self.metrics["South pole temp"].config(text=f"{state['south_pole_c']:.2f} °C")

        if self.finished:
            status = "Finished"
        elif self.paused:
            status = "Paused"
        else:
            status = f"Running: {elapsed:.1f} / {self.duration_seconds:.0f} s"
        self.metrics["Animation status"].config(text=status)

    def _tick(self) -> None:
        if self.paused:
            return

        elapsed = self.elapsed_real_seconds()
        if elapsed >= self.duration_seconds:
            self.finished = True

        self.draw_frame()

        if not self.finished:
            self.root.after(FPS_MS, self._tick)

    def run(self) -> None:
        self._tick()
        self.root.mainloop()


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def choose_selected_planet(planets: list[Exoplanet], requested_choice: int | None) -> Exoplanet:
    by_choice = {planet.menu_index: planet for planet in planets}

    if requested_choice is not None:
        if requested_choice not in by_choice:
            raise ValueError(f"Choice {requested_choice} was not found. Choose a number from 1 to {len(planets)}.")
        return by_choice[requested_choice]

    if len(planets) == 1 and planets[0].name == "Earth" and planets[0].host == "Sun":
        print("No exoplanet CSV was supplied, so the program will use Earth orbiting the Sun.")
        return planets[0]

    print_planet_menu(planets)
    while True:
        raw = input(f"Enter a planet number from 1 to {len(planets)}: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if choice in by_choice:
            return by_choice[choice]
        print(f"That choice was not found. Enter a number from 1 to {len(planets)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate exoplanet orbital temperature from a CSV file.")
    parser.add_argument(
        "csv_file",
        type=Path,
        nargs="?",
        default=None,
        help="Optional CSV file containing exoplanet data. If omitted, Earth/Sun is used.",
    )
    parser.add_argument(
        "--choice",
        type=int,
        default=None,
        help="Planet menu number to choose directly. This is 1 to n, not the CSV row index.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Deprecated alias for --choice. Use --choice instead.",
    )
    parser.add_argument("--single", action="store_true", help="Animate only the selected planet, not the full host system")
    parser.add_argument("--years", type=float, default=10_000.0, help="Simulated years covered by the animation")
    parser.add_argument("--duration", type=float, default=60.0, help="Real seconds the animation should run")
    parser.add_argument("--tilt", type=float, default=None, help="Override axial tilt in degrees for all planets")
    args = parser.parse_args()

    requested_choice = args.choice if args.choice is not None else args.row

    if args.csv_file is None:
        planets = [earth_sun_planet()]
    else:
        if not args.csv_file.exists():
            print(f"Could not find CSV file: {args.csv_file}")
            print("Using the built-in Earth/Sun fallback instead.")
            planets = [earth_sun_planet()]
        else:
            planets = load_planets(args.csv_file, forced_tilt_deg=args.tilt)
            if not planets:
                print("No usable exoplanets were loaded from the CSV.")
                print("Using the built-in Earth/Sun fallback instead.")
                planets = [earth_sun_planet()]

    selected = choose_selected_planet(planets, requested_choice)

    if args.single:
        displayed_planets = [selected]
    else:
        displayed_planets = [planet for planet in planets if planet.host == selected.host]
        if not displayed_planets:
            displayed_planets = [selected]

    print(f"\nSelected: {selected.name} around {selected.host}")
    print(f"Animating {len(displayed_planets)} planet(s) for {args.duration:.0f} seconds.")
    print(f"Simulation span: {args.years:,.0f} years.")
    print("The host star is fixed at the centre of the animation.\n")

    app = ExoplanetOrbitApp(
        planets=displayed_planets,
        selected=selected,
        simulated_years=args.years,
        duration_seconds=args.duration,
    )
    app.run()


if __name__ == "__main__":
    main()
