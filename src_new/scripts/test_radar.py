"""
Test radar.py end-to-end against your real DEM and SHARAD orbit data.

Wires together: sharad.py (geometry) -> dem.py (terrain) -> facets.py
(facet row) -> radar.py (range/incidence/power) for one along-track trace,
and checks the results are physically sensible.

Edit the two paths below, then run:
    python test_radar.py
"""
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
import numpy as np
import matplotlib.pyplot as plt

from dem import MarsDEM
from sharad import SHARADOrbit
from facets import generate_facet_row
from radar import (
    spacecraft_to_local,
    compute_range_and_incidence,
    compute_received_power,
    apply_incidence_angle_cutoff,
    normalize_power_db,
)

# ---------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------
DEM_PATH = PROJECT_DIR / 'data' / 'Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'

SHARAD_BASE_PATH = PROJECT_DIR / 'data' / 'sharad' / 's_00571601'

TEST_TRACE_INDEX = 750


def main():
    print("Loading DEM...")
    dem = MarsDEM(DEM_PATH)

    print("Loading SHARAD orbit...")
    orbit = SHARADOrbit(SHARAD_BASE_PATH)

    idx = TEST_TRACE_INDEX
    nadir_lat = orbit.lat[idx]
    nadir_lon = orbit.lon[idx]
    heading = orbit.heading[idx]
    sc_radius = orbit.sc_radius[idx]
    nadir_radius = orbit.mars_radius[idx]

    print(f"\nTrace {idx}: lat={nadir_lat:.4f}, lon={nadir_lon:.4f}, "
          f"heading={heading:.2f} deg")
    print(f"GEOM.TAB altitude (sc_radius - mars_radius): "
          f"{sc_radius - nadir_radius:.1f} m")

    # -------------------------------------------------------------
    # Step 1: generate facet row
    # -------------------------------------------------------------
    print("\nGenerating facet row...")
    row = generate_facet_row(
        nadir_lat, nadir_lon, heading, dem,
        cross_track_extent=45000.0,
        facet_size_cross=30.0,
        facet_size_along=300.0,
    )
    n_facets = len(row['cross_track_dist'])
    print(f"  {n_facets} facets generated")
    if n_facets == 0:
        print("No facets - can't continue. Check DEM coverage at this location.")
        return

    # -------------------------------------------------------------
    # Step 2: spacecraft position in the same local frame
    # -------------------------------------------------------------
    e_hat = orbit.e_hat[idx]
    n_hat = orbit.n_hat[idx]
    r_hat = orbit.r_hat[idx]
    sc_mbfc = orbit.spacecraft_positions[idx]
    nadir_mbfc = orbit.nadir_positions[idx]

    sc_local = spacecraft_to_local(
        sc_mbfc, nadir_mbfc, e_hat, n_hat, r_hat
    )
    print(f"\nSpacecraft in local frame (east, north, up): {sc_local}")
    print(f"  'up' should equal GEOM.TAB altitude: {sc_radius - nadir_radius:.1f} m")

    # -------------------------------------------------------------
    # Step 3: range + incidence angle for every facet
    # -------------------------------------------------------------
    facet_centers_local = np.column_stack([
        row['cross_track_dist'] * np.sin(np.radians(heading) + np.pi / 2),
        row['cross_track_dist'] * np.cos(np.radians(heading) + np.pi / 2),
        row['center_elev'],
    ])

    ranges, inc_deg, cos_inc = compute_range_and_incidence(
        facet_centers_local, row['normal_local'], sc_local
    )

    print(f"\nRange: min={ranges.min()/1000:.2f} km, max={ranges.max()/1000:.2f} km")
    print(f"  (nadir facet range should be close to altitude: "
          f"{sc_radius - nadir_radius:.1f} m)")
    print(f"Incidence angle: min={inc_deg.min():.2f} deg, max={inc_deg.max():.2f} deg")

    nadir_facet_idx = np.argmin(np.abs(row['cross_track_dist']))
    print(f"  At cross_track~0 (index {nadir_facet_idx}): "
          f"range={ranges[nadir_facet_idx]:.1f} m, "
          f"incidence={inc_deg[nadir_facet_idx]:.2f} deg")

    # -------------------------------------------------------------
    # Step 4: power
    # -------------------------------------------------------------
    powers = compute_received_power(row['area'], ranges, cos_inc)
    powers = apply_incidence_angle_cutoff(powers, inc_deg, max_angle=85.0)

    n_zeroed = np.sum(powers == 0)
    print(f"\nPower: {n_zeroed}/{n_facets} facets zeroed by 85 deg cutoff")
    print(f"  Peak power at cross_track={row['cross_track_dist'][np.argmax(powers)]/1000:.2f} km "
          f"(expect near 0, i.e. near nadir, for smooth terrain)")

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(row['cross_track_dist'] / 1000, row['center_elev'])
    axes[0].set_ylabel('Elevation\nrel. to nadir (m)')
    axes[0].set_title(f'Trace {idx}: lat={nadir_lat:.2f}, lon={nadir_lon:.2f}')
    axes[0].grid(alpha=0.3)

    axes[1].plot(row['cross_track_dist'] / 1000, ranges / 1000)
    axes[1].axhline((sc_radius - nadir_radius) / 1000, color='r', linestyle='--',
                     label='spacecraft altitude')
    axes[1].set_ylabel('Range (km)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(row['cross_track_dist'] / 1000, inc_deg)
    axes[2].set_ylabel('Incidence (deg)')
    axes[2].grid(alpha=0.3)

    powers_db = 10 * np.log10(powers + 1e-30)
    axes[3].plot(row['cross_track_dist'] / 1000, powers_db)
    axes[3].set_ylabel('Power (dB, arbitrary)')
    axes[3].set_xlabel('Cross-track distance (km)')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('radar_test.png', dpi=120)
    print("\nSaved plot: radar_test.png")


if __name__ == '__main__':
    main()