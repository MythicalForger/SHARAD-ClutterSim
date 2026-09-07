"""
Test cluttergram.py: run the full simulation for a real orbit and compare
against the actual SHARAD radargram, side by side.

This runs the complete pipeline (every trace, ~1500+, each generating a
facet row of ~3000 facets), so it will take real time - expect on the
order of minutes depending on your machine, not seconds.

Edit the two paths below, then run:
    python test_cluttergram.py
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
from cluttergram import generate_cluttergram

# ---------------------------------------------------------------------
# EDIT THESE PATHS
# ---------------------------------------------------------------------
DEM_PATH = PROJECT_DIR / 'data' / 'Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'

SHARAD_BASE_PATH = PROJECT_DIR / 'data' / 'sharad' / 's_00571601'


def main():
    print("Loading DEM...")
    dem = MarsDEM(DEM_PATH)

    print("Loading SHARAD orbit...")
    orbit = SHARADOrbit(SHARAD_BASE_PATH)

    print("\nGenerating cluttergram (this will take a while)...\n")
    cluttergram = generate_cluttergram(orbit, dem, verbose=True)

    print("\nLoading real radargram for comparison...")
    real_radargram = orbit.read_radargram()
    print(f"  Real radargram shape: {real_radargram.shape}")
    print(f"  Simulated cluttergram shape: {cluttergram.data.shape}")

    sim_norm = cluttergram.normalize(db_range=80.0)

    real_safe = np.abs(real_radargram) + 1e-30
    real_db = 10 * np.log10(real_safe)
    real_db_max = np.max(real_db)
    real_db_min = real_db_max - 80.0
    real_clipped = np.clip(real_db, real_db_min, real_db_max)
    real_norm = (255 * (real_clipped - real_db_min) / 80.0).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(sim_norm.T, cmap='gray', aspect='auto', origin='upper')
    axes[0].set_title('Simulated Cluttergram (surface-only)')
    axes[0].set_xlabel('Along-track position')
    axes[0].set_ylabel('Time delay (bin)')

    axes[1].imshow(real_norm, cmap='gray', aspect='auto', origin='upper')
    axes[1].set_title('Real SHARAD Radargram')
    axes[1].set_xlabel('Along-track trace')
    axes[1].set_ylabel('Sample (time)')

    plt.tight_layout()
    plt.savefig('cluttergram_comparison.png', dpi=120)
    print("\nSaved: cluttergram_comparison.png")

    cluttergram.save('cluttergram_test.npz')


if __name__ == '__main__':
    main()