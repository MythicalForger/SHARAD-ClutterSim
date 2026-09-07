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

# ---------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------
DEM_PATH = PROJECT_DIR / 'data' / 'Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'

SHARAD_BASE_PATH = PROJECT_DIR / 'data' / 'sharad' / 's_00571601'

# Which along-track trace index to generate a facet row for (0 = orbit start)
TEST_TRACE_INDEX = 750   # roughly midway through the 1537-trace orbit


def main():
    print("Loading DEM...")
    dem = MarsDEM(DEM_PATH)

    print("\nLoading SHARAD orbit...")
    orbit = SHARADOrbit(SHARAD_BASE_PATH)
    info = orbit.get_orbit_info()
    print(f"  Traces: {info['n_traces']}, Samples: {info['n_samples']}")
    print(f"  Lat range: {info['lat_range']}")
    print(f"  Lon range: {info['lon_range']}")
    print(f"  Altitude: {info['altitude_mean']/1000:.1f} km "
          f"({info['altitude_range'][0]/1000:.1f}-{info['altitude_range'][1]/1000:.1f} km)")

    idx = TEST_TRACE_INDEX
    nadir_lat = orbit.lat[idx]
    nadir_lon = orbit.lon[idx]
    heading = orbit.heading[idx]

    print(f"\nGenerating facet row at trace {idx}:")
    print(f"  Nadir: lat={nadir_lat:.4f}, lon={nadir_lon:.4f}")
    print(f"  Heading: {heading:.2f} deg")

    # Sanity: what does the DEM say the elevation is right at nadir?
    nadir_elev = dem.get_elevation(nadir_lat, nadir_lon)
    print(f"  DEM elevation at nadir: {nadir_elev:.1f} m")
    print(f"  GEOM.TAB mars_radius at this trace: {orbit.mars_radius[idx]:.1f} m "
          f"(reference radius {3396190.0:.0f} m -> implies elev "
          f"{orbit.mars_radius[idx] - 3396190.0:.1f} m)")

    row = generate_facet_row(
        nadir_lat, nadir_lon, heading, dem,
        cross_track_extent=45000.0,
        facet_size_cross=30.0,
        facet_size_along=300.0,
    )

    n_facets = len(row['cross_track_dist'])
    print(f"\nFacets generated: {n_facets} (out of {int(2*45000/30)+1} possible)")

    if n_facets == 0:
        print("No valid facets - check DEM path/coverage at this location.")
        return

    print(f"  Cross-track range: {row['cross_track_dist'].min()/1000:.1f} to "
          f"{row['cross_track_dist'].max()/1000:.1f} km")
    print(f"  Elevation range: {row['center_elev'].min():.1f} to "
          f"{row['center_elev'].max():.1f} m")
    print(f"  Area range: {row['area'].min():.1f} to {row['area'].max():.1f} m^2 "
          f"(flat expectation: {30*300} m^2)")
    print(f"  Normal z-component range: {row['normal_local'][:,2].min():.4f} to "
          f"{row['normal_local'][:,2].max():.4f} (1.0 = perfectly flat/vertical)")

    all_unit = np.allclose(np.linalg.norm(row['normal_local'], axis=1), 1.0)
    print(f"  All normals unit length: {all_unit}")

    # -------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(row['cross_track_dist'] / 1000, row['center_elev'])
    axes[0].set_ylabel('Elevation (m)')
    axes[0].set_title(f'Facet row at trace {idx} (lat={nadir_lat:.2f}, lon={nadir_lon:.2f})')
    axes[0].grid(alpha=0.3)

    axes[1].plot(row['cross_track_dist'] / 1000, row['area'])
    axes[1].axhline(30 * 300, color='r', linestyle='--', label='flat expectation')
    axes[1].set_ylabel('Facet area (m^2)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    incidence_from_vertical = np.degrees(np.arccos(np.clip(row['normal_local'][:, 2], -1, 1)))
    axes[2].plot(row['cross_track_dist'] / 1000, incidence_from_vertical)
    axes[2].set_ylabel('Tilt from vertical (deg)')
    axes[2].set_xlabel('Cross-track distance (km)')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('facet_row_test.png', dpi=120)
    print("\nSaved plot: facet_row_test.png")


if __name__ == '__main__':
    main()