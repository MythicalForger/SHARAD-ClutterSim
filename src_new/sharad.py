import csv
import numpy as np
from pathlib import Path

MARS_RADIUS = 3396190.0 #metres


class SHARADOrbit:
    """
    Loads one SHARAD orbit's radargram + geometry.

    Usage:
        orbit = SHARADOrbit('/path/to/s_00571601')
        radargram = orbit.read_radargram()          # (n_samples, n_traces)
        sc_pos = orbit.spacecraft_positions          # (n_traces, 3) MBFC meters
        sc_vel = orbit.spacecraft_velocities         # (n_traces, 3) MBFC m/s
        nadir_pos = orbit.nadir_positions            # (n_traces, 3) MBFC meters
    """

    def __init__(self, base_path):
        self.base_path = Path(base_path)

        self.rgram_img = Path(str(self.base_path) + '_rgram.img')
        self.rgram_lbl = Path(str(self.base_path) + '_rgram.lbl.txt')
        self.geom_tab = Path(str(self.base_path) + '_geom.tab.txt')

        if not self.rgram_img.exists():
            raise FileNotFoundError(f"Radargram IMG not found: {self.rgram_img}")
        if not self.geom_tab.exists():
            raise FileNotFoundError(f"GEOM.TAB not found: {self.geom_tab}")

        self._read_rgram_label()
        self._read_geom_tab()
        self._build_geometry()

        # if self.n_geom_records != self.n_traces:
        #     print(f"Warning: GEOM.TAB has {self.n_geom_records} records but "
        #           f"rgram has {self.n_traces} traces - check these match "
        #           f"before trusting per-trace geometry.")

    def _read_rgram_label(self):
        text = self.rgram_lbl.read_text(errors='ignore')
        self.n_samples = int(_extract_label_value(text, 'LINES'))
        self.n_traces = int(_extract_label_value(text, 'LINE_SAMPLES'))

    def read_radargram(self):
        data = np.fromfile(self.rgram_img, dtype='<f4')
        return data.reshape(self.n_samples, self.n_traces)

    def _read_geom_tab(self):
        cols = {
            'lat': [], 'lon': [], 'mars_radius': [], 'sc_radius': [],
            'radial_vel': [], 'tangential_vel': [],
        }

        with open(self.geom_tab, newline='') as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                cols['lat'].append(float(row[2]))
                cols['lon'].append(float(row[3]))
                cols['mars_radius'].append(float(row[4]) * 1000.0)  
                cols['sc_radius'].append(float(row[5]) * 1000.0)    
                cols['radial_vel'].append(float(row[6]))             
                cols['tangential_vel'].append(float(row[7]))         

        self.lat = np.array(cols['lat'])
        self.lon = np.array(cols['lon'])
        self.mars_radius = np.array(cols['mars_radius'])
        self.sc_radius = np.array(cols['sc_radius'])
        self.radial_vel = np.array(cols['radial_vel'])
        self.tangential_vel = np.array(cols['tangential_vel'])
        self.n_geom_records = len(self.lat)

    def _build_geometry(self):
        lat_rad = np.radians(self.lat)
        lon_rad = np.radians(self.lon)

        cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
        cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)

        r_hat = np.column_stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
        n_hat = np.column_stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        e_hat = np.column_stack([-sin_lon, cos_lon, np.zeros_like(lon_rad)])

        self.spacecraft_positions = self.sc_radius[:, None] * r_hat

        self.nadir_positions = self.mars_radius[:, None] * r_hat # not yet DEM-corrected.

        # Ground-track heading (bearing) at each trace, from consecutive
        # lat/lon pairs. Last point repeats the previous heading.
        heading = _bearing(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        heading = np.append(heading, heading[-1])
        heading_rad = np.radians(heading)

        tangent_dir = (
            np.cos(heading_rad)[:, None] * n_hat +
            np.sin(heading_rad)[:, None] * e_hat
        )

        self.spacecraft_velocities = (
            self.radial_vel[:, None] * r_hat +
            self.tangential_vel[:, None] * tangent_dir
        )

        self.r_hat = r_hat
        self.n_hat = n_hat
        self.e_hat = e_hat
        self.heading = heading

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_orbit_info(self):
        altitude = self.sc_radius - self.mars_radius
        return {
            'n_traces': self.n_traces,
            'n_samples': self.n_samples,
            'n_geom_records': self.n_geom_records,
            'lat_range': (self.lat.min(), self.lat.max()),
            'lon_range': (self.lon.min(), self.lon.max()),
            'altitude_mean': altitude.mean(),
            'altitude_range': (altitude.min(), altitude.max()),
        }


def _extract_label_value(text, key):
    import re
    match = re.search(rf'{key}\s*=\s*(\d+)', text)
    if not match:
        raise ValueError(f"Could not find {key} in label")
    return match.group(1)


def _bearing(lat1, lon1, lat2, lon2):
    """
    Initial bearing (degrees, 0-360, 0=north) from point 1 to point 2
    on a sphere. Standard great-circle bearing formula.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360