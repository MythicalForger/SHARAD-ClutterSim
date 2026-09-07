"""
Cluttergram assembly: runs facets.py + radar.py across an entire orbit and
bins the resulting per-facet power into a 2D (along-track, time-delay) image.

Time delay is measured relative to the nadir echo, matching how real SHARAD
radargrams are referenced (see sharad's Reduced Data Record documentation:
round-trip time delay on the vertical axis, along-track on the horizontal).
"""

import numpy as np
import time
from pathlib import Path

from facets import generate_facet_row
from radar import (
    spacecraft_to_local,
    compute_range_and_incidence,    
    compute_received_power,
    apply_incidence_angle_cutoff,
)

SPEED_OF_LIGHT = 299792458.0      # m/s
SAMPLE_INTERVAL = 37.5e-9         # seconds, SHARAD's per-bin time resolution
MAX_TIME_DELAY = 135e-6           # seconds after nadir - matches 3600-sample rgram


def range_to_relative_time_delay(range_facet, range_nadir, speed_of_light=SPEED_OF_LIGHT):
    """
    Two-way time delay of a facet's echo, relative to the nadir echo.

    delta_tau = 2*(R_facet - R_nadir) / c

    Facets farther than nadir arrive later (positive delay) - this is what
    gets plotted on a radargram's Y-axis, not absolute range.
    """
    return 2.0 * (range_facet - range_nadir) / speed_of_light

def time_delay_to_bin(time_delay, sample_interval=SAMPLE_INTERVAL):
    """Floor time delay (seconds) to an integer bin index."""
    return np.floor(time_delay / sample_interval).astype(int)

class Cluttergram:
    """
    Accumulator for simulated clutter power, shape (n_positions, n_time_bins).

    Mirrors a real SHARAD radargram's layout: columns are along-track
    positions/traces, rows are time-delay bins after the nadir echo.
    """

    def __init__(self, n_positions, n_time_bins=None, max_time_delay=MAX_TIME_DELAY,
                 sample_interval=SAMPLE_INTERVAL):
        if n_time_bins is None:
            n_time_bins = int(max_time_delay / sample_interval)

        self.n_positions = n_positions
        self.n_time_bins = n_time_bins
        self.sample_interval = sample_interval
        self.max_time_delay = max_time_delay

        self.data = np.zeros((n_positions, n_time_bins), dtype=np.float64)

    def accumulate_power(self, position_index, time_delays, powers):
        """
        Add facet powers into the correct time-delay bins for one column
        (along-track position). Facets landing in the same bin sum
        incoherently (P_total = P1 + P2 + ...).
        """
        bin_indices = time_delay_to_bin(time_delays, self.sample_interval)

        valid = (bin_indices >= 0) & (bin_indices < self.n_time_bins)
        bin_indices = bin_indices[valid]
        powers_valid = powers[valid]

        np.add.at(self.data[position_index, :], bin_indices, powers_valid)

    def normalize(self, db_range=80.0):
        """Normalize to [0, 255] uint8 on a dB scale, for visualization."""
        data_safe = self.data + 1e-30
        data_db = 10 * np.log10(data_safe)
        db_max = np.max(data_db)
        db_min = db_max - db_range
        clipped = np.clip(data_db, db_min, db_max)
        normalized = (clipped - db_min) / db_range
        return (255 * normalized).astype(np.uint8)

    def save(self, filepath):
        filepath = Path(filepath)
        np.savez_compressed(
            filepath,
            data=self.data,
            n_positions=self.n_positions,
            n_time_bins=self.n_time_bins,
            sample_interval=self.sample_interval,
            max_time_delay=self.max_time_delay,
        )
        print(f"Cluttergram saved: {filepath}")

    @classmethod
    def load(cls, filepath):
        loaded = np.load(Path(filepath))
        cg = cls(
            n_positions=int(loaded['n_positions']),
            n_time_bins=int(loaded['n_time_bins']),
            max_time_delay=float(loaded['max_time_delay']),
            sample_interval=float(loaded['sample_interval']),
        )
        cg.data = loaded['data']
        return cg


def generate_cluttergram(orbit, dem,
                          cross_track_extent=45000.0,
                          facet_size_cross=30.0,
                          facet_size_along=300.0,
                          max_incidence_angle=85.0,
                          verbose=True):
    """
    Run the full facets -> radar -> binning pipeline across every trace in
    an orbit, producing a simulated cluttergram.

    Parameters
    ----------
    orbit : SHARADOrbit
        Loaded orbit (sharad.py), provides per-trace lat/lon/heading/radii
        and MBFC positions + local frame vectors.
    dem : MarsDEM
        Loaded DEM (dem.py), used by facets.py for terrain elevation.
    cross_track_extent, facet_size_cross, facet_size_along : float
        Facet row generation parameters, passed through to facets.py.
    max_incidence_angle : float
        Facets beyond this incidence angle contribute zero power.
    verbose : bool
        Print progress.

    Returns
    -------
    Cluttergram
    """
    n_positions = orbit.n_geom_records
    cluttergram = Cluttergram(n_positions)

    if verbose:
        print(f"Generating cluttergram for {n_positions} positions...")
        print(f"  Time bins: {cluttergram.n_time_bins}, "
              f"max delay: {cluttergram.max_time_delay*1e6:.1f} us")

    start_time = time.time()
    progress_interval = max(1, n_positions // 20)
    total_facets = 0

    for i in range(n_positions):
        nadir_lat = orbit.lat[i]
        nadir_lon = orbit.lon[i]
        heading = orbit.heading[i]
        sc_radius = orbit.sc_radius[i]
        nadir_radius = orbit.mars_radius[i]

        row = generate_facet_row(
            nadir_lat, nadir_lon, heading, dem,
            cross_track_extent, facet_size_cross, facet_size_along,
        )

        n_facets = len(row['cross_track_dist'])
        if n_facets == 0:
            continue
        total_facets += n_facets

        # Spacecraft position in the same local frame as this facet row
        e_hat, n_hat, r_hat = orbit.e_hat[i], orbit.n_hat[i], orbit.r_hat[i]
        sc_mbfc = orbit.spacecraft_positions[i]
        nadir_mbfc = orbit.nadir_positions[i]
        sc_local = spacecraft_to_local(
            sc_mbfc, nadir_mbfc, e_hat, n_hat, r_hat
        )

        heading_rad = np.radians(heading)
        facet_centers_local = np.column_stack([
            row['cross_track_dist'] * np.sin(heading_rad + np.pi / 2),
            row['cross_track_dist'] * np.cos(heading_rad + np.pi / 2),
            row['center_elev'],
        ])

        ranges, inc_deg, cos_inc = compute_range_and_incidence(
            facet_centers_local, row['normal_local'], sc_local
        )

        powers = compute_received_power(row['area'], ranges, cos_inc)
        powers = apply_incidence_angle_cutoff(powers, inc_deg, max_incidence_angle)

        # Range to nadir itself: the facet at (or nearest) cross_track=0,
        # using the spacecraft's own local 'up' distance as the nadir range
        # (spacecraft is at local (0,0,altitude), nadir is at local (0,0,0),
        # so range_nadir = altitude = sc_local[2] exactly).
        range_nadir = sc_local[2]

        time_delays = range_to_relative_time_delay(ranges, range_nadir)
        cluttergram.accumulate_power(i, time_delays, powers)

        if verbose and (i % progress_interval == 0 or i == n_positions - 1):
            elapsed = time.time() - start_time
            progress = (i + 1) / n_positions
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  {i+1:5d}/{n_positions} ({progress*100:5.1f}%) | "
                  f"facets: {n_facets:4d} | elapsed: {elapsed:6.1f}s | ETA: {eta:6.1f}s")

    if verbose:
        elapsed = time.time() - start_time
        print(f"\nDone in {elapsed:.1f}s. Total facets: {total_facets:,}")
        nonzero = np.sum(cluttergram.data > 0)
        print(f"Fill rate: {nonzero:,}/{cluttergram.data.size:,} "
              f"({100*nonzero/cluttergram.data.size:.2f}%)")

    return cluttergram