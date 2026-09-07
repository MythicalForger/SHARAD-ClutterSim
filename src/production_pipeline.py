"""
Production pipeline for SHARAD clutter simulation.

End-to-end workflow:
1. Load SHARAD orbit data
2. Process DEM corridor
3. Generate cluttergram (streaming)
4. Compare with real radargram
5. Generate echo map
"""

import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

from sharad_reader import SHARADReader
from dem_processing import process_dem_for_orbit
from cluttergram import generate_cluttergram, Cluttergram
import config

def run_full_simulation(sharad_base_path, mola_file, 
                       output_dir,
                       subsample_factor=1,
                       cross_track_extent=45000,
                       facet_size_cross=30.0,
                       facet_size_along=300.0,
                       verbose=True):
    """
    Run complete clutter simulation for SHARAD orbit.
    
    Parameters:
    -----------
    sharad_base_path : str or Path
        Base path to SHARAD files (e.g., 's_00571601_rgram')
    mola_file : str or Path
        Path to MOLA DEM file
    output_dir : str or Path
        Directory for outputs
    subsample_factor : int
        Subsample factor for positions (1 = all, 2 = every other, etc.)
    cross_track_extent : float
        Half-width of swath (meters)
    facet_size_cross, facet_size_along : float
        Facet dimensions (meters)
    verbose : bool
        Print progress
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - cluttergram
        - radargram (real data)
        - reader (SHARAD reader object)
        - interpolator (DEM interpolator)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*70)
        print("SHARAD CLUTTER SIMULATION - PRODUCTION PIPELINE")
        print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD SHARAD DATA
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: LOADING SHARAD ORBIT DATA")
        print("="*70)
    
    reader = SHARADReader(sharad_base_path)
    
    # Get orbit info
    orbit_info = reader.get_orbit_info()
    
    if verbose:
        print(f"\nOrbit information:")
        print(f"  Orbit number: {orbit_info['orbit_number']}")
        print(f"  Records: {orbit_info['n_records']}")
        print(f"  Latitude: {orbit_info['lat_range'][0]:.2f}° to "
              f"{orbit_info['lat_range'][1]:.2f}°")
        print(f"  Longitude: {orbit_info['lon_range'][0]:.2f}° to "
              f"{orbit_info['lon_range'][1]:.2f}°")
        print(f"  Altitude: {orbit_info['altitude_mean']/1000:.1f} km "
              f"({orbit_info['altitude_range'][0]/1000:.1f} - "
              f"{orbit_info['altitude_range'][1]/1000:.1f} km)")
    
    # Get positions
    spacecraft_positions = reader.get_spacecraft_positions()
    nadir_positions = reader.get_nadir_positions()
    velocities = reader.get_spacecraft_velocities()
    lats, lons = reader.get_ground_track()
    
    # Subsample if requested
    if subsample_factor > 1:
        if verbose:
            print(f"\nSubsampling by factor {subsample_factor}...")
        
        indices = np.arange(0, len(spacecraft_positions), subsample_factor)
        spacecraft_positions = spacecraft_positions[indices]
        nadir_positions = nadir_positions[indices]
        velocities = velocities[indices]
        lats = lats[indices]
        lons = lons[indices]
        
        if verbose:
            print(f"  Reduced to {len(spacecraft_positions)} positions")
    
    # ========================================================================
    # STEP 2: PROCESS DEM
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: PROCESSING DEM")
        print("="*70)
    
    interpolator, elevation, lat_array, lon_array = process_dem_for_orbit(
        mola_file=mola_file,
        ground_track_lat=lats,
        ground_track_lon=lons,
        apply_filter=True,
        filter_sigma=2.0,
        detect_outliers=True
    )
    
    # ========================================================================
    # STEP 3: GENERATE CLUTTERGRAM
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: GENERATING CLUTTERGRAM")
        print("="*70)
    
    start_time = time.time()
    
    cluttergram = generate_cluttergram(
        spacecraft_positions=spacecraft_positions,
        nadir_positions=nadir_positions,
        velocities=velocities,
        dem_interpolator=interpolator,
        cross_track_extent=cross_track_extent,
        facet_size_cross=facet_size_cross,
        facet_size_along=facet_size_along,
        apply_angle_cutoff=True,
        max_incidence_angle=85.0,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nCluttergram generation complete in {elapsed:.1f} seconds")
    
    # Save cluttergram
    cluttergram_file = output_dir / f"cluttergram_orbit_{orbit_info['orbit_number']}.npz"
    cluttergram.save(cluttergram_file)
    
    # ========================================================================
    # STEP 4: LOAD REAL RADARGRAM
    # ========================================================================
    
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: LOADING REAL RADARGRAM")
        print("="*70)
    
    radargram = reader.read_radargram()
    
    # Subsample radargram to match cluttergram
    if subsample_factor > 1:
        radargram = radargram[indices, :]
    
    if verbose:
        print(f"  Radargram shape: {radargram.shape}")
        print(f"  Cluttergram shape: {cluttergram.data.shape}")
    
    # ========================================================================
    # RETURN RESULTS
    # ========================================================================
    
    results = {
        'cluttergram': cluttergram,
        'radargram': radargram,
        'reader': reader,
        'interpolator': interpolator,
        'elevation': elevation,
        'lat_array': lat_array,
        'lon_array': lon_array,
        'spacecraft_positions': spacecraft_positions,
        'nadir_positions': nadir_positions,
        'orbit_info': orbit_info,
    }
    
    return results


def compare_cluttergram_radargram(cluttergram, radargram, output_file=None):
    """
    Create side-by-side comparison of simulated vs real data.
    
    Parameters:
    -----------
    cluttergram : Cluttergram
        Simulated cluttergram
    radargram : ndarray
        Real SHARAD radargram
    output_file : str or Path, optional
        Save figure to file
    
    Returns:
    --------
    fig : matplotlib figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Normalize cluttergram
    clutter_norm = cluttergram.normalize(method='dB', db_range=80)
    
    # Normalize radargram (dB scale)
    radargram_safe = np.abs(radargram) + 1e-30
    radargram_db = 10 * np.log10(radargram_safe)
    radargram_db_max = np.max(radargram_db)
    radargram_db_min = radargram_db_max - 80
    radargram_db_clip = np.clip(radargram_db, radargram_db_min, radargram_db_max)
    radargram_norm = ((radargram_db_clip - radargram_db_min) / 80 * 255).astype(np.uint8)
    
    # Simulated cluttergram
    ax1 = axes[0]
    im1 = ax1.imshow(clutter_norm.T, cmap='gray', aspect='auto', origin='upper')
    ax1.set_xlabel('Along-track Position', fontsize=12)
    ax1.set_ylabel('Time Delay', fontsize=12)
    ax1.set_title('Simulated Cluttergram', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Power (dB)')
    
    # Real radargram
    ax2 = axes[1]
    im2 = ax2.imshow(radargram_norm.T, cmap='gray', aspect='auto', origin='upper')
    ax2.set_xlabel('Along-track Position', fontsize=12)
    ax2.set_ylabel('Sample Number', fontsize=12)
    ax2.set_title('Real SHARAD Radargram', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='Power (dB)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison saved: {output_file}")
    
    return fig


def generate_echo_map(cluttergram, spacecraft_positions, nadir_positions,
                     output_file=None):
    """
    Generate echo map (geographic visualization of clutter).
    
    Shows where surface clutter originates geographically.
    
    Parameters:
    -----------
    cluttergram : Cluttergram
        Cluttergram data
    spacecraft_positions : ndarray
        Spacecraft positions
    nadir_positions : ndarray
        Nadir positions
    output_file : str or Path, optional
        Save to file
    
    Returns:
    --------
    fig : matplotlib figure
    """
    
    from coordinates import mbfc_to_geodetic
    
    # For echo map, we need to track which geographic locations
    # contribute to clutter at each position
    
    # This is a simplified version - full version would track
    # individual facet contributions
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract ground track
    ground_lats = []
    ground_lons = []
    
    for nadir in nadir_positions:
        lat, lon, _ = mbfc_to_geodetic(nadir[0], nadir[1], nadir[2])
        ground_lats.append(lat)
        ground_lons.append(lon)
    
    ground_lats = np.array(ground_lats)
    ground_lons = np.array(ground_lons)
    
    # Compute total power per position
    power_per_position = cluttergram.data.sum(axis=1)
    
    # Normalize for visualization
    power_norm = (power_per_position - power_per_position.min())
    power_norm = power_norm / power_norm.max() if power_norm.max() > 0 else power_norm
    
    # Plot
    scatter = ax.scatter(ground_lons, ground_lats, 
                        c=power_norm, cmap='hot', s=20, alpha=0.6)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title('Echo Map (Clutter Intensity)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046)
    cbar.set_label('Normalized Clutter Power', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Echo map saved: {output_file}")
    
    return fig