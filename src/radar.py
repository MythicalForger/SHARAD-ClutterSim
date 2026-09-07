"""
Radar power calculations for SHARAD clutter simulation.

Implements paper's Equations (1-4):
- Antenna gain
- Radar cross-section  
- Effective area
- Received power

Plus normalization and utilities.
"""

import numpy as np
from config import WAVELENGTH, SPEED_OF_LIGHT
from coordinates import compute_range, compute_incidence_angle

# ============================================================================
# RADAR EQUATION COMPONENTS
# ============================================================================

def compute_effective_area(facet_area, cos_incidence):
    """
    Compute effective area of facet (Equation 4).
    
    A_eff = A_facet × cos(θ_i)
    
    Parameters:
    -----------
    facet_area : float or ndarray
        Physical area of facet in m²
    cos_incidence : float or ndarray
        Cosine of incidence angle (0 to 1)
    
    Returns:
    --------
    effective_area : float or ndarray
        Effective illuminated area in m²
    
    Physical meaning:
    -----------------
    Facet tilted at angle θ appears smaller to radar.
    
    Example:
        facet_area = 9000 m²
        θ_i = 0° (face-on) → cos(0°) = 1.0 → A_eff = 9000 m²
        θ_i = 60° (tilted) → cos(60°) = 0.5 → A_eff = 4500 m²
        θ_i = 90° (edge-on) → cos(90°) = 0.0 → A_eff = 0 m²
    
    Why this formula:
    -----------------
    Project facet area onto plane perpendicular to radar look direction.
    This is standard geometric projection: A_projected = A × cos(angle)
    
    Code notes:
    -----------
    Works with scalars or arrays (NumPy broadcasting).
    No validation here - assumes cos_incidence already in [0, 1].
    """
    
    return facet_area * cos_incidence


def compute_antenna_gain(effective_area, wavelength=WAVELENGTH):
    """
    Compute antenna gain (Equation 2).
    
    G = 4π × A_eff / λ²
    
    Parameters:
    -----------
    effective_area : float or ndarray
        Effective area in m²
    wavelength : float
        Radar wavelength in meters (default from config)
    
    Returns:
    --------
    gain : float or ndarray
        Antenna gain (dimensionless)
    
    Physical meaning:
    -----------------
    Gain measures how well antenna focuses energy vs isotropic radiator.
    G = 1: radiates equally in all directions (isotropic)
    G > 1: focuses energy in preferred direction
    
    For SHARAD dipole: G ≈ 1.5 (slight focusing)
    
    Why 4π/λ²?
    ----------
    Comes from electromagnetic theory (antenna reciprocity).
    Larger aperture → higher gain (focus energy more).
    Smaller wavelength → higher gain (better focusing).
    
    Paper's interpretation:
    -----------------------
    Uses facet effective area instead of antenna area.
    This is modeling convenience - combines target geometry
    with antenna characteristics. Valid for relative power
    calculations (which is what we need for clutter).
    
    Code notes:
    -----------
    np.pi is NumPy's constant for π (3.14159...)
    Result is dimensionless (ratio of power densities)
    """
    
    return 4 * np.pi * effective_area / (wavelength**2)


def compute_radar_cross_section(effective_area, wavelength=WAVELENGTH):
    """
    Compute radar cross-section (Equation 3).
    
    σ = 4π × A²_eff / λ²
    
    Parameters:
    -----------
    effective_area : float or ndarray
        Effective area in m²
    wavelength : float
        Radar wavelength in meters
    
    Returns:
    --------
    rcs : float or ndarray
        Radar cross-section in m²
    
    Physical meaning:
    -----------------
    RCS is effective area that would scatter same power isotropically.
    
    For diffuse scatterer (Lambertian):
        σ ∝ A² (not just A!)
    
    Why A²?
    -------
    Optical regime (target >> wavelength):
    - Target intercepts power ∝ A
    - Target re-radiates ∝ A (radiating aperture)
    - Combined: σ ∝ A × A = A²
    
    Why 4π/λ²?
    ----------
    Same reasoning as gain - from EM theory.
    This factor ensures dimensionally correct (m²).
    
    Code notes:
    -----------
    effective_area**2 is element-wise squaring for arrays.
    Returns RCS in square meters.
    """
    
    return 4 * np.pi * effective_area**2 / (wavelength**2)


def compute_received_power(facet_area, range_m, cos_incidence, 
                          wavelength=WAVELENGTH):
    """
    Compute received power from a facet (Equation 1).
    
    P_r = G² × λ² × σ / (64π³ × R⁴)
    
    Combining equations (2), (3), (4):
    P_r = [4π·A_eff/λ²]² × λ² × [4π·A²_eff/λ²] / (64π³ × R⁴)
    
    Simplifies to:
    P_r ∝ A⁴_eff / R⁴
    P_r ∝ A⁴_facet × cos⁴(θ_i) / R⁴
    
    Parameters:
    -----------
    facet_area : float or ndarray
        Facet area in m²
    range_m : float or ndarray
        Range from spacecraft to facet in meters
    cos_incidence : float or ndarray
        Cosine of incidence angle
    wavelength : float
        Radar wavelength in meters
    
    Returns:
    --------
    power : float or ndarray
        Received power (arbitrary units, relative)
    
    Physical interpretation:
    ------------------------
    Power ∝ A⁴:
        Double facet size → 16× power (!)
        This is because both gain and RCS ∝ A
    
    Power ∝ cos⁴(θ_i):
        Face-on (0°): cos⁴(0) = 1.0 (100% power)
        30° tilt: cos⁴(30°) = 0.56 (56% power)
        45° tilt: cos⁴(45°) = 0.25 (25% power)
        60° tilt: cos⁴(60°) = 0.0625 (6% power!)
        Very strong falloff!
    
    Power ∝ 1/R⁴:
        Range increases 10% → power drops 34%
        Range doubles → power drops by factor 16
        Two-way path amplifies distance losses
    
    Transmitted power ignored:
    --------------------------
    We only care about RELATIVE power between facets.
    Transmitted power P_t is same for all → cancels
    in normalization. Same for receiving antenna area.
    
    Code walkthrough:
    -----------------
    Step 1: Compute effective area from facet area and incidence
    Step 2: Compute gain from effective area
    Step 3: Compute RCS from effective area  
    Step 4: Combine via Friis equation
    
    All steps vectorized - works on arrays efficiently.
    """
    
    # Step 1: Effective area (Equation 4)
    A_eff = compute_effective_area(facet_area, cos_incidence)
    
    # Step 2: Antenna gain (Equation 2)
    G = compute_antenna_gain(A_eff, wavelength)
    
    # Step 3: Radar cross-section (Equation 3)
    sigma = compute_radar_cross_section(A_eff, wavelength)
    
    # Step 4: Received power (Equation 1)
    # Denominator: 64 × π³ × R⁴
    denominator = 64 * (np.pi**3) * (range_m**4)
    
    # Numerator: G² × λ² × σ
    numerator = (G**2) * (wavelength**2) * sigma
    
    # Received power
    power = numerator / denominator
    
    return power


def compute_received_power_simple(facet_area, range_m, cos_incidence):
    """
    Simplified power calculation using derived formula.
    
    Skips intermediate steps, directly computes:
    P_r ∝ A⁴ × cos⁴(θ_i) / R⁴
    
    This is mathematically equivalent to compute_received_power()
    but faster (fewer operations, no intermediate arrays).
    
    Use this for production code.
    Use compute_received_power() for validation/understanding.
    
    Returns power in arbitrary units (constants omitted).
    """
    
    # Combined formula from algebraic simplification
    power = (facet_area**4) * (cos_incidence**4) / (range_m**4)
    
    return power


# ============================================================================
# BATCH POWER CALCULATION
# ============================================================================

def compute_powers_for_facets(facet_properties, spacecraft_pos):
    """
    Compute received power for array of facets (vectorized).
    
    This is the main function you'll use in practice.
    Takes facet properties (from facets.py) and spacecraft position,
    returns array of powers.
    
    Parameters:
    -----------
    facet_properties : dict
        From facets.compute_facet_properties_batch()
        Must contain:
        - 'centers': (n, 3) array of positions
        - 'normals': (n, 3) array of normals
        - 'areas': (n,) array of areas
    spacecraft_pos : ndarray, shape (3,)
        Spacecraft position in MBFC
    
    Returns:
    --------
    powers : ndarray, shape (n,)
        Received power from each facet (arbitrary units)
    ranges : ndarray, shape (n,)
        Range to each facet (meters)
    incidence_angles : ndarray, shape (n,)
        Incidence angle for each facet (degrees)
    cos_incidences : ndarray, shape (n,)
        Cosine of incidence angles
    
    Algorithm:
    ----------
    1. Compute ranges (vectorized)
    2. Compute incidence angles (vectorized)
    3. Compute powers (vectorized)
    
    All operations use NumPy broadcasting for speed.
    
    Performance:
    ------------
    For 3000 facets: ~1 ms (vectorized)
    vs ~300 ms (Python loop)
    300× speedup!
    """
    
    # Extract facet data
    centers = facet_properties['centers']      # (n_facets, 3)
    normals = facet_properties['normals']      # (n_facets, 3)
    areas = facet_properties['areas']          # (n_facets,)
    
    n_facets = len(areas)
    
    # Step 1: Compute ranges (vectorized)
    # Broadcasting: (3,) - (n, 3) → (n, 3)
    diff_vectors = spacecraft_pos - centers
    ranges = np.linalg.norm(diff_vectors, axis=1)  # (n,)
    
    # Step 2: Compute incidence angles (vectorized)
    # Normalize vectors
    look_vectors = diff_vectors / ranges[:, np.newaxis]  # (n, 3)
    normals_unit = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]  # (n, 3)
    
    # Dot product for each facet
    # Element-wise multiply, then sum along rows
    cos_incidences = np.abs(np.sum(look_vectors * normals_unit, axis=1))  # (n,)
    
    # Clip to valid range (handle numerical errors)
    cos_incidences = np.clip(cos_incidences, 0, 1)
    
    # Convert to angles for reference
    incidence_angles = np.rad2deg(np.arccos(cos_incidences))  # (n,)
    
    # Step 3: Compute powers (vectorized)
    # Use simple formula for speed
    powers = compute_received_power_simple(areas, ranges, cos_incidences)
    
    return powers, ranges, incidence_angles, cos_incidences


# ============================================================================
# POWER NORMALIZATION
# ============================================================================

def normalize_power_db(powers, db_range=80.0):
    """
    Normalize power to [0, 255] using dB scale.
    
    Converts linear power to decibels, then scales to 8-bit range.
    This is standard for radar data visualization.
    
    Parameters:
    -----------
    powers : ndarray
        Linear power values (arbitrary units)
    db_range : float
        Dynamic range to display in dB (default 80 dB)
    
    Returns:
    --------
    normalized : ndarray, dtype uint8
        Normalized values in [0, 255]
    
    Algorithm:
    ----------
    1. Convert to dB: P_dB = 10 × log₁₀(P)
    2. Find maximum: P_dB_max
    3. Set minimum: P_dB_min = P_dB_max - db_range
    4. Clip to range: [P_dB_min, P_dB_max]
    5. Normalize to [0, 1]: (P_dB - P_dB_min) / db_range
    6. Scale to [0, 255]
    
    Why dB scale?
    -------------
    Radar data has huge dynamic range (10⁸ or 80 dB).
    Human eye can only distinguish ~2 orders of magnitude.
    Log scale compresses range while preserving features.
    
    Why 80 dB?
    ----------
    Standard for radar visualization.
    Shows both strong (near) and weak (far) echoes.
    Adjustable based on data characteristics.
    
    Code details:
    -------------
    powers + 1e-30: Avoid log(0) = -inf
    np.log10: Base-10 logarithm (dB uses log₁₀, not ln)
    np.clip: Clamp values to range
    astype(np.uint8): Convert float to 8-bit integer
    """
    
    # Avoid log(0) by adding tiny value
    powers_safe = powers + 1e-30
    
    # Convert to dB
    # 10 × log₁₀(P) is standard dB formula for power
    powers_db = 10 * np.log10(powers_safe)
    
    # Find max (reference level)
    p_db_max = np.max(powers_db)
    
    # Set min based on dynamic range
    p_db_min = p_db_max - db_range
    
    # Clip to range
    powers_db_clipped = np.clip(powers_db, p_db_min, p_db_max)
    
    # Normalize to [0, 1]
    normalized_float = (powers_db_clipped - p_db_min) / db_range
    
    # Scale to [0, 255] and convert to uint8
    normalized = (255 * normalized_float).astype(np.uint8)
    
    return normalized


def normalize_power_linear(powers):
    """
    Normalize power to [0, 255] using linear scale.
    
    Simple min-max normalization.
    Use for testing, but dB is better for real data.
    
    Formula:
        normalized = 255 × (P - P_min) / (P_max - P_min)
    """
    
    p_min = np.min(powers)
    p_max = np.max(powers)
    
    # Avoid division by zero
    if p_max == p_min:
        return np.full_like(powers, 128, dtype=np.uint8)
    
    normalized_float = (powers - p_min) / (p_max - p_min)
    normalized = (255 * normalized_float).astype(np.uint8)
    
    return normalized


def normalize_power_sqrt(powers):
    """
    Normalize power using square root scale.
    
    Intermediate between linear and log.
    
    Formula:
        normalized = 255 × (√P - √P_min) / (√P_max - √P_min)
    
    Compresses dynamic range less aggressively than dB.
    """
    
    # Ensure non-negative
    powers_safe = np.maximum(powers, 0)
    
    powers_sqrt = np.sqrt(powers_safe)
    
    p_min = np.min(powers_sqrt)
    p_max = np.max(powers_sqrt)
    
    if p_max == p_min:
        return np.full_like(powers, 128, dtype=np.uint8)
    
    normalized_float = (powers_sqrt - p_min) / (p_max - p_min)
    normalized = (255 * normalized_float).astype(np.uint8)
    
    return normalized


# ============================================================================
# FILTERING AND THRESHOLDING
# ============================================================================

def apply_incidence_angle_cutoff(powers, incidence_angles, max_angle=85.0):
    """
    Set power to zero for facets beyond incidence angle cutoff.
    
    Facets nearly edge-on contribute negligible power and add noise.
    Standard practice: ignore facets with θ_i > 85°.
    
    Parameters:
    -----------
    powers : ndarray
        Power values
    incidence_angles : ndarray
        Incidence angles in degrees
    max_angle : float
        Maximum acceptable incidence angle (degrees)
    
    Returns:
    --------
    powers_filtered : ndarray
        Powers with high-angle facets zeroed
    
    Why 85°?
    --------
    cos(85°) = 0.087
    cos⁴(85°) = 0.000057
    Power is < 0.006% of face-on value!
    
    Negligible contribution, but adds:
    - Computational overhead
    - Numerical noise
    - Clutter from numerical errors
    
    Code notes:
    -----------
    Creates copy (doesn't modify input).
    Boolean indexing: powers[mask] selects elements where mask is True.
    """
    
    powers_filtered = powers.copy()
    
    # Create mask: True where angle exceeds cutoff
    high_angle_mask = incidence_angles > max_angle
    
    # Zero out high-angle facets
    powers_filtered[high_angle_mask] = 0.0
    
    # Report how many were filtered
    n_filtered = np.sum(high_angle_mask)
    if n_filtered > 0:
        fraction = n_filtered / len(powers)
        # Only print if significant (> 1%)
        if fraction > 0.01:
            print(f"  Filtered {n_filtered} facets ({fraction*100:.1f}%) "
                  f"with θ_i > {max_angle}°")
    
    return powers_filtered


def apply_range_weighting(powers, ranges, reference_range=None):
    """
    Apply range-dependent weighting (optional).
    
    Sometimes useful to emphasize near-nadir returns.
    
    Formula:
        P_weighted = P × (R_ref / R)^α
    
    where α controls weighting strength.
    
    Not used in standard clutter simulation,
    but included for completeness.
    """
    
    if reference_range is None:
        reference_range = np.min(ranges)
    
    # Power law weighting (α = 2 gives R² weighting)
    alpha = 2.0
    weights = (reference_range / ranges)**alpha
    
    powers_weighted = powers * weights
    
    return powers_weighted


# ============================================================================
# DIAGNOSTICS AND VALIDATION
# ============================================================================

def compute_power_statistics(powers, ranges, incidence_angles, verbose=True):
    """
    Compute diagnostic statistics for power calculations.
    
    Useful for validation and debugging.
    
    Returns dict with:
    - Power statistics (min, max, mean, etc.)
    - Range statistics
    - Incidence angle statistics
    - Power vs range correlation
    - Power vs angle correlation
    """
    
    # Remove zeros for statistics (filtered facets)
    nonzero_mask = powers > 0
    powers_nz = powers[nonzero_mask]
    ranges_nz = ranges[nonzero_mask]
    angles_nz = incidence_angles[nonzero_mask]
    
    if len(powers_nz) == 0:
        return {'error': 'All powers are zero'}
    
    stats = {
        # Power statistics
        'n_facets_total': len(powers),
        'n_facets_nonzero': len(powers_nz),
        'power_min': np.min(powers_nz),
        'power_max': np.max(powers_nz),
        'power_mean': np.mean(powers_nz),
        'power_std': np.std(powers_nz),
        'power_median': np.median(powers_nz),
        
        # Dynamic range
        'dynamic_range_linear': np.max(powers_nz) / np.min(powers_nz),
        'dynamic_range_db': 10 * np.log10(np.max(powers_nz) / np.min(powers_nz)),
        
        # Range statistics
        'range_min_km': np.min(ranges_nz) / 1000,
        'range_max_km': np.max(ranges_nz) / 1000,
        'range_mean_km': np.mean(ranges_nz) / 1000,
        
        # Incidence angle statistics
        'angle_min_deg': np.min(angles_nz),
        'angle_max_deg': np.max(angles_nz),
        'angle_mean_deg': np.mean(angles_nz),
        
        # Correlations (for validation)
        'power_range_correlation': np.corrcoef(powers_nz, ranges_nz)[0, 1],
        'power_angle_correlation': np.corrcoef(powers_nz, angles_nz)[0, 1],
    }
    
    if verbose:
        print("\nPower Calculation Statistics:")
        print(f"  Facets: {stats['n_facets_nonzero']}/{stats['n_facets_total']}")
        print(f"  Power range: {stats['power_min']:.2e} to {stats['power_max']:.2e}")
        print(f"  Dynamic range: {stats['dynamic_range_db']:.1f} dB")
        print(f"  Range: {stats['range_min_km']:.2f} to {stats['range_max_km']:.2f} km")
        print(f"  Incidence: {stats['angle_min_deg']:.2f}° to {stats['angle_max_deg']:.2f}°")
        print(f"  Power-Range correlation: {stats['power_range_correlation']:.3f} (expect < -0.9)")
        print(f"  Power-Angle correlation: {stats['power_angle_correlation']:.3f} (expect < -0.9)")
    
    return stats


def validate_power_dependencies(facet_area, range_m, incidence_deg):
    """
    Validate that power scales correctly with parameters.
    
    Tests:
    1. Power ∝ 1/R⁴ (hold area, angle constant)
    2. Power ∝ A⁴ (hold range, angle constant)  
    3. Power ∝ cos⁴(θ) (hold range, area constant)
    
    Returns True if all tests pass.
    """
    
    # Test 1: Range dependence
    ranges = np.array([300000, 310000, 320000])  # 300, 310, 320 km
    cos_inc = np.cos(np.deg2rad(10))  # 10° incidence
    
    powers_range = compute_received_power_simple(facet_area, ranges, cos_inc)
    
    # Check R⁴ scaling
    expected_ratio = (ranges[0] / ranges[1])**4
    actual_ratio = powers_range[0] / powers_range[1]
    
    range_test_pass = abs(actual_ratio - expected_ratio) < 0.01
    
    # Test 2: Area dependence
    areas = np.array([5000, 9000, 15000])  # Different areas
    powers_area = compute_received_power_simple(areas, 300000, cos_inc)
    
    expected_ratio = (areas[1] / areas[0])**4
    actual_ratio = powers_area[1] / powers_area[0]
    
    area_test_pass = abs(actual_ratio - expected_ratio) < 0.01
    
    # Test 3: Angle dependence
    angles = np.array([0, 30, 60])  # degrees
    cos_angles = np.cos(np.deg2rad(angles))
    powers_angle = compute_received_power_simple(facet_area, 300000, cos_angles)
    
    expected_ratio = (cos_angles[1] / cos_angles[0])**4
    actual_ratio = powers_angle[1] / powers_angle[0]
    
    angle_test_pass = abs(actual_ratio - expected_ratio) < 0.01
    
    if range_test_pass and area_test_pass and angle_test_pass:
        print("✓ All power dependency tests passed!")
        return True
    else:
        print("✗ Power dependency tests FAILED:")
        if not range_test_pass:
            print(f"  Range test: expected {expected_ratio:.6f}, got {actual_ratio:.6f}")
        if not area_test_pass:
            print(f"  Area test: failed")
        if not angle_test_pass:
            print(f"  Angle test: failed")
        return False