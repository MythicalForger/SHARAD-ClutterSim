"""
Facet generation for SHARAD clutter simulation.

Creates rectangular surface facets:
- 30m × 300m size
- Positioned along spacecraft ground track
- ±45 km cross-track extent
- With 3D positions, normals, areas
"""

import numpy as np
from pathlib import Path

from coordinates import (
    geodetic_to_mbfc,
    compute_local_frame,
    compute_surface_normal,
    compute_facet_area,
    cross_track_distance
)
from dem_processing import BilinearInterpolator
import config

# ============================================================================
# FACET DATA STRUCTURE
# ============================================================================

class Facet:
    """
    Container for facet geometric properties.
    
    Why a class?
    ------------
    - Groups related data (position, normal, area)
    - Easier to pass around than multiple arrays
    - Can add methods later (e.g., plotting, validation)
    - More readable than dict or tuple
    
    What we store:
    --------------
    - center: 3D position in MBFC (X, Y, Z)
    - corners: 4 corner positions in MBFC (for visualization)
    - normal: perpendicular vector (for incidence angle)
    - area: surface area in m² (for power equation)
    - lat, lon: geographic coordinates (for reference)
    - cross_track_dist: signed distance from ground track (for L/R separation)
    
    Why not just NumPy structured array?
    ------------------------------------
    - Less intuitive: facet['center'] vs facet.center
    - Harder to add methods
    - Class is more Pythonic for this use case
    """
    
    def __init__(self):
        """
        Initialize empty facet.
        
        All attributes set to None initially.
        Will be filled during facet generation.
        """
        self.center = None          # (X, Y, Z) in MBFC
        self.corners = None         # 4×3 array of corner positions
        self.normal = None          # (nx, ny, nz) normal vector
        self.area = None            # m²
        self.lat = None             # degrees
        self.lon = None             # degrees
        self.cross_track_dist = None  # meters (signed)
    
    def __repr__(self):
        """
        String representation for debugging.
        
        Called when you print(facet) or use repr(facet).
        Useful for inspecting facet properties.
        """
        if self.center is None:
            return "Facet(uninitialized)"
        
        return (f"Facet(lat={self.lat:.3f}°, lon={self.lon:.3f}°, "
                f"area={self.area:.1f}m², cross_track={self.cross_track_dist/1000:.1f}km)")


# ============================================================================
# FACET GENERATION - SINGLE POSITION
# ============================================================================

def generate_facets_at_position(spacecraft_pos, nadir_pos, velocity, 
                                dem_interpolator, 
                                cross_track_extent=45000,
                                facet_size_cross=30.0,
                                facet_size_along=300.0):
    """
    Generate facets for a single spacecraft position.
    
    Creates a cross-track row of rectangular facets.
    
    Parameters:
    -----------
    spacecraft_pos : ndarray, shape (3,)
        Spacecraft position in MBFC (meters)
    nadir_pos : ndarray, shape (3,)
        Nadir point position in MBFC (meters)
    velocity : ndarray, shape (3,)
        Spacecraft velocity in MBFC (m/s)
    dem_interpolator : BilinearInterpolator
        Initialized DEM interpolator
    cross_track_extent : float
        Half-width of swath in meters (default ±45 km)
    facet_size_cross : float
        Facet size in cross-track direction (meters)
    facet_size_along : float
        Facet size in along-track direction (meters)
    
    Returns:
    --------
    facets : list of Facet
        Generated facets
    
    Algorithm overview:
    -------------------
    1. Compute local coordinate frame (A, C, Z vectors)
    2. Generate cross-track positions (±45 km)
    3. For each position:
        a. Compute 4 corner positions in local frame
        b. Convert to geographic (lat, lon)
        c. Get elevations from DEM interpolator
        d. Convert to 3D MBFC with elevations
        e. Compute facet properties (normal, area, center)
    4. Return list of Facet objects
    
    Why local frame?
    ----------------
    Much easier to work in "forward/right/up" coordinates
    than global MBFC. We compute in local frame, then transform.
    
    Code walkthrough:
    -----------------
    """
    
    # Step 1: Compute local coordinate frame at spacecraft position
    # A_hat = along-track (forward)
    # C_hat = cross-track (right)  
    # Z_hat = vertical (up)
    A_hat, C_hat, Z_hat = compute_local_frame(spacecraft_pos, nadir_pos, velocity)
    
    # Step 2: Generate cross-track positions
    # From -45 km to +45 km in 30m steps
    # np.arange: start, stop, step
    # Why not include endpoint? Because we want facet centers, not edges
    cross_track_distances = np.arange(
        -cross_track_extent, 
        cross_track_extent + facet_size_cross,  # +facet_size ensures we include endpoint
        facet_size_cross
    )
    
    n_facets = len(cross_track_distances)
    
    # Initialize facet list
    # Pre-allocate for efficiency (could also use list.append)
    facets = []
    
    # Step 3: Loop over cross-track positions
    for i, d_cross in enumerate(cross_track_distances):
        
        # Create a facet object
        facet = Facet()
        
        # Store cross-track distance
        facet.cross_track_dist = d_cross
        
        # Step 3a: Compute 4 corner positions in local frame
        # Corners arranged as:
        #   1 ---- 2
        #   |      |
        #   4 ---- 3
        
        # Half-sizes for offsetting from center
        half_cross = facet_size_cross / 2
        half_along = facet_size_along / 2
        
        # Corner offsets in local frame (cross_track, along_track, vertical)
        # Vertical=0 initially; will be set by DEM elevation
        corner_offsets = np.array([
            [d_cross - half_cross, -half_along, 0],  # Corner 1 (lower-left)
            [d_cross + half_cross, -half_along, 0],  # Corner 2 (lower-right)
            [d_cross + half_cross, +half_along, 0],  # Corner 3 (upper-right)
            [d_cross - half_cross, +half_along, 0],  # Corner 4 (upper-left)
        ])
        
        # Step 3b: Convert corner offsets to global MBFC positions (at nadir elevation)
        # Transform: local → global using basis vectors
        # Position = nadir + offset_cross*C_hat + offset_along*A_hat + offset_vert*Z_hat
        corners_mbfc_flat = []
        
        for offset in corner_offsets:
            # Linear combination of basis vectors
            position_local = (offset[0] * C_hat + 
                            offset[1] * A_hat + 
                            offset[2] * Z_hat)
            position_global = nadir_pos + position_local
            corners_mbfc_flat.append(position_global)
        
        # Convert list to array for easier manipulation
        corners_mbfc_flat = np.array(corners_mbfc_flat)  # Shape: (4, 3)
        
        # Step 3c: Convert MBFC corners to geodetic to query DEM
        from coordinates import mbfc_to_geodetic
        
        corner_lats = []
        corner_lons = []
        
        for corner in corners_mbfc_flat:
            lat, lon, _ = mbfc_to_geodetic(corner[0], corner[1], corner[2])
            corner_lats.append(lat)
            corner_lons.append(lon)
        
        corner_lats = np.array(corner_lats)
        corner_lons = np.array(corner_lons)
        
        # Step 3d: Get elevations from DEM interpolator
        # Vectorized call: get all 4 elevations at once
        corner_elevations = dem_interpolator(corner_lats, corner_lons)
        
        # Check for NaN elevations (outside DEM bounds or missing data)
        if np.any(~np.isfinite(corner_elevations)):
            # Skip this facet (outside valid DEM region)
            continue
        
        # Step 3e: Update corner positions with actual elevations
        # Convert (lat, lon, elevation) back to MBFC
        corners_mbfc = []
        
        for lat, lon, elev in zip(corner_lats, corner_lons, corner_elevations):
            X, Y, Z = geodetic_to_mbfc(lat, lon, elev)
            corners_mbfc.append([X, Y, Z])
        
        corners_mbfc = np.array(corners_mbfc)  # Shape: (4, 3)
        
        # Store corners
        facet.corners = corners_mbfc
        
        # Step 3f: Compute facet center
        # Simple average of 4 corners
        facet.center = np.mean(corners_mbfc, axis=0)
        
        # Convert center to geodetic for reference
        facet.lat, facet.lon, _ = mbfc_to_geodetic(
            facet.center[0], facet.center[1], facet.center[2]
        )
        
        # Step 3g: Compute facet normal
        # Using corner points (from Section 2 coordinates.py)
        facet.normal = compute_surface_normal(
            corners_mbfc[0], corners_mbfc[1], 
            corners_mbfc[2], corners_mbfc[3]
        )
        
        # Step 3h: Compute facet area
        facet.area = compute_facet_area(
            corners_mbfc[0], corners_mbfc[1], 
            corners_mbfc[2], corners_mbfc[3]
        )
        
        # Step 3i: Validation checks
        # Check if area is reasonable
        expected_area = facet_size_cross * facet_size_along
        if facet.area < 0.1 * expected_area or facet.area > 10 * expected_area:
            # Something went wrong (degenerate facet, bad DEM data)
            continue
        
        # Check if normal magnitude is reasonable
        normal_magnitude = np.linalg.norm(facet.normal)
        if normal_magnitude < 1e-6:
            # Degenerate facet (nearly zero area)
            continue
        
        # Add to list
        facets.append(facet)
    
    return facets


# ============================================================================
# FACET GENERATION - FULL ORBIT
# ============================================================================

def generate_facets_for_orbit(spacecraft_positions, nadir_positions, velocities,
                               dem_interpolator,
                               cross_track_extent=45000,
                               facet_size_cross=30.0,
                               facet_size_along=300.0,
                               verbose=True):
    """
    Generate facets for entire orbit.
    
    Loops over all spacecraft positions and generates facets.
    
    Parameters:
    -----------
    spacecraft_positions : ndarray, shape (n_positions, 3)
        Spacecraft positions in MBFC
    nadir_positions : ndarray, shape (n_positions, 3)
        Nadir positions in MBFC
    velocities : ndarray, shape (n_positions, 3)
        Velocity vectors in MBFC
    dem_interpolator : BilinearInterpolator
        DEM interpolator
    cross_track_extent : float
        Half-width of swath (meters)
    facet_size_cross, facet_size_along : float
        Facet dimensions (meters)
    verbose : bool
        Print progress
    
    Returns:
    --------
    all_facets : list of lists
        all_facets[i] = list of facets for position i
    
    Why list of lists?
    ------------------
    - Each position has different number of facets (some skipped due to NaN)
    - Irregular structure doesn't fit into rectangular array
    - List of lists is flexible
    
    Alternative: Could flatten to single list with position indices
    
    Memory consideration:
    ---------------------
    For typical orbit:
    - 12,000 positions
    - 3,000 facets per position
    - 36 million facets total
    - Each facet ~500 bytes → 18 GB if storing everything
    
    Solution: Don't store all facets!
    - Generate on-the-fly during cluttergram assembly
    - Only store current position's facets
    - This function exists for testing/visualization
    - In production (Section 7), we'll use streaming approach
    """
    
    n_positions = len(spacecraft_positions)
    
    if verbose:
        print(f"Generating facets for {n_positions} positions...")
        print(f"  Cross-track extent: ±{cross_track_extent/1000:.1f} km")
        print(f"  Facet size: {facet_size_cross}m × {facet_size_along}m")
    
    all_facets = []
    
    # Progress tracking
    progress_interval = max(1, n_positions // 20)  # Update every 5%
    
    for i in range(n_positions):
        
        # Get parameters for this position
        sc_pos = spacecraft_positions[i]
        nadir_pos = nadir_positions[i]
        vel = velocities[i]
        
        # Generate facets
        facets = generate_facets_at_position(
            sc_pos, nadir_pos, vel, dem_interpolator,
            cross_track_extent, facet_size_cross, facet_size_along
        )
        # Debug: check local frame orthogonality
        from coordinates import compute_local_frame
        
        all_facets.append(facets)
        
        # Progress update
        if verbose and (i % progress_interval == 0 or i == n_positions - 1):
            n_facets_this_pos = len(facets)
            total_facets_so_far = sum(len(f) for f in all_facets)
            progress = (i + 1) / n_positions * 100
            print(f"  Progress: {progress:5.1f}% | "
                  f"Position {i+1}/{n_positions} | "
                  f"Facets this pos: {n_facets_this_pos} | "
                  f"Total so far: {total_facets_so_far}")
    
    # Summary statistics
    if verbose:
        total_facets = sum(len(f) for f in all_facets)
        avg_facets = total_facets / n_positions
        
        print(f"\nFacet generation complete:")
        print(f"  Total facets: {total_facets:,}")
        print(f"  Average per position: {avg_facets:.1f}")
        
        # Memory estimate
        bytes_per_facet = 500  # Rough estimate
        memory_mb = total_facets * bytes_per_facet / 1024**2
        print(f"  Estimated memory: {memory_mb:.1f} MB")
    
    return all_facets


# ============================================================================
# FACET PROPERTIES - BATCH COMPUTATION
# ============================================================================

def compute_facet_properties_batch(facets):
    """
    Extract facet properties into arrays for vectorized operations.
    
    Converts list of Facet objects into NumPy arrays.
    Useful for vectorized calculations (ranges, angles, powers).
    
    Parameters:
    -----------
    facets : list of Facet
        Facets from a single position
    
    Returns:
    --------
    properties : dict
        Dictionary with arrays:
        - 'centers': (n_facets, 3) positions
        - 'normals': (n_facets, 3) normal vectors
        - 'areas': (n_facets,) areas in m²
        - 'lats': (n_facets,) latitudes
        - 'lons': (n_facets,) longitudes
        - 'cross_track_dists': (n_facets,) signed distances
    
    Why convert to arrays?
    ----------------------
    NumPy vectorization is 100× faster than Python loops.
    
    Example usage:
        props = compute_facet_properties_batch(facets)
        ranges = np.linalg.norm(spacecraft_pos - props['centers'], axis=1)
    
    Much faster than:
        ranges = [np.linalg.norm(spacecraft_pos - f.center) for f in facets]
    """
    
    n_facets = len(facets)
    
    # Pre-allocate arrays
    centers = np.zeros((n_facets, 3))
    normals = np.zeros((n_facets, 3))
    areas = np.zeros(n_facets)
    lats = np.zeros(n_facets)
    lons = np.zeros(n_facets)
    cross_track_dists = np.zeros(n_facets)
    
    # Fill arrays
    # Could use list comprehension + np.array, but loop is clearer
    for i, facet in enumerate(facets):
        centers[i] = facet.center
        normals[i] = facet.normal
        areas[i] = facet.area
        lats[i] = facet.lat
        lons[i] = facet.lon
        cross_track_dists[i] = facet.cross_track_dist
    
    return {
        'centers': centers,
        'normals': normals,
        'areas': areas,
        'lats': lats,
        'lons': lons,
        'cross_track_dists': cross_track_dists
    }


# ============================================================================
# FACET FILTERING
# ============================================================================

def filter_facets_by_criteria(facets, 
                               min_area=None,
                               max_area=None,
                               max_cross_track_dist=None):
    """
    Filter facets based on quality criteria.
    
    Useful for removing:
    - Degenerate facets (very small area)
    - Outliers (unreasonably large area)
    - Far off-nadir facets (low power, not interesting)
    
    Parameters:
    -----------
    facets : list of Facet
        Input facets
    min_area : float, optional
        Minimum acceptable area (m²)
    max_area : float, optional
        Maximum acceptable area (m²)
    max_cross_track_dist : float, optional
        Maximum cross-track distance (meters)
    
    Returns:
    --------
    filtered_facets : list of Facet
        Facets passing all criteria
    
    Example:
        # Keep only facets within ±30 km and reasonable areas
        good_facets = filter_facets_by_criteria(
            facets,
            min_area=500,      # At least 500 m²
            max_area=100000,   # At most 100,000 m²
            max_cross_track_dist=30000  # Within ±30 km
        )
    """
    
    filtered = []
    
    for facet in facets:
        
        # Area checks
        if min_area is not None and facet.area < min_area:
            continue
        
        if max_area is not None and facet.area > max_area:
            continue
        
        # Cross-track distance check
        if max_cross_track_dist is not None:
            if abs(facet.cross_track_dist) > max_cross_track_dist:
                continue
        
        # Passed all checks
        filtered.append(facet)
    
    return filtered


# ============================================================================
# UTILITIES
# ============================================================================

def get_facet_statistics(facets):
    """
    Compute statistics for a list of facets.
    
    Useful for validation and quality checking.
    
    Returns dict with:
    - n_facets: count
    - area_mean, area_std, area_min, area_max
    - cross_track_range
    - lat_range, lon_range
    """
    
    if not facets:
        return {'n_facets': 0}
    
    areas = np.array([f.area for f in facets])
    cross_tracks = np.array([f.cross_track_dist for f in facets])
    lats = np.array([f.lat for f in facets])
    lons = np.array([f.lon for f in facets])
    
    stats = {
        'n_facets': len(facets),
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'area_min': np.min(areas),
        'area_max': np.max(areas),
        'cross_track_min': np.min(cross_tracks),
        'cross_track_max': np.max(cross_tracks),
        'lat_min': np.min(lats),
        'lat_max': np.max(lats),
        'lon_min': np.min(lons),
        'lon_max': np.max(lons),
    }
    
    return stats


def visualize_facet_grid(facets, spacecraft_pos, nadir_pos, elev_min=-2000, elev_max=3000):
    """
    Create 3D visualization of facet grid.
    
    Shows:
    - Facet positions and orientations
    - Spacecraft location
    - Color-coded by elevation
    
    Useful for debugging facet generation.
    
    Note: Only visualize small number of facets (< 1000)
    or plot will be very slow!
    """
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract elevations for coloring
    elevations = np.array([
        # Get Z-component of center (height above ellipsoid)
        f.center[2] for f in facets
    ])
    
    # Normalize for colormap
    norm = plt.Normalize(vmin=elev_min, vmax=elev_max)
    cmap = plt.cm.terrain
    
    # Plot facets as polygons
    for facet in facets:
        # Get corner positions (convert to km for visualization)
        corners = facet.corners / 1000  # Convert m → km
        
        # Create polygon (close the loop)
        verts = [corners[[0, 1, 2, 3, 0]]]
        
        # Color by elevation
        elev = facet.center[2]
        color = cmap(norm(elev))
        
        poly = Poly3DCollection(verts, facecolors=color, 
                               edgecolors='k', linewidths=0.1, alpha=0.7)
        ax.add_collection3d(poly)
    
    # Plot spacecraft
    sc_km = spacecraft_pos / 1000
    ax.scatter(*sc_km, color='red', s=100, marker='^', 
              label='Spacecraft', depthshade=False)
    
    # Plot nadir
    nadir_km = nadir_pos / 1000
    ax.scatter(*nadir_km, color='green', s=100, marker='o', 
              label='Nadir', depthshade=False)
    
    # Set labels and limits
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Facet Grid Visualization', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio
    # Get bounds
    all_coords = np.vstack([f.corners for f in facets]) / 1000
    max_range = np.array([
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
        all_coords[:, 2].max() - all_coords[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    return fig, ax