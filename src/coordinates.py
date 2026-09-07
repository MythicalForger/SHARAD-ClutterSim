import numpy as np
from config import (
    MARS_EQUATORIAL_RADIUS,
    MARS_POLAR_RADIUS,
    MARS_FLATTENING,
    MARS_ECCENTRICITY_SQUARED
)

#Convert geodetic (lat/lon/height) to Cartesian (X, Y, Z) coordinates
def geodetic_to_mbfc(lat, lon, height):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    N = MARS_EQUATORIAL_RADIUS / np.sqrt(1 - MARS_ECCENTRICITY_SQUARED * np.sin(lat_rad)**2)
    X = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    # The (1 - e²) factor accounts for polar flattening
    Z = (N * (1 - MARS_ECCENTRICITY_SQUARED) + height) * np.sin(lat_rad)
    return X, Y, Z

#Convert Cartesian (X, Y, Z) to geodetic (lat/lon/height) coordinates
def mbfc_to_geodetic(X, Y, Z, tolerance=1e-6, max_iterations=10):
    lon = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)

    lat = np.arctan2(Z, p)
    for i in range(max_iterations):
        lat_old = lat
        N = MARS_EQUATORIAL_RADIUS / np.sqrt(1 - MARS_ECCENTRICITY_SQUARED * np.sin(lat)**2)
        lat = np.arctan2(Z + MARS_ECCENTRICITY_SQUARED * N * np.sin(lat), p)
        if np.abs(lat - lat_old) < tolerance:
            break

    N = MARS_EQUATORIAL_RADIUS / np.sqrt(1 - MARS_ECCENTRICITY_SQUARED * np.sin(lat)**2)
    height = p / np.cos(lat) - N
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    # Normalize longitude to [0, 360) range
    lon_deg = lon_deg % 360
    
    return lat_deg, lon_deg, height

# CROSS-TRACK COORDINATE SYSTEM
def compute_local_frame(spacecraft_pos, nadir_pos, velocity):
    vertical = spacecraft_pos - nadir_pos
    Z_hat = vertical / np.linalg.norm(vertical)

    # Gram-Schmidt: remove the Z component from velocity so A_hat ⊥ Z_hat
    vel_unit = velocity / np.linalg.norm(velocity)
    vel_ortho = vel_unit - np.dot(vel_unit, Z_hat) * Z_hat
    vel_ortho_mag = np.linalg.norm(vel_ortho)

    if vel_ortho_mag < 1e-6:
        # Velocity is nearly radial — pick least-aligned global axis as fallback
        for ref in [np.array([0.,0.,1.]), np.array([1.,0.,0.]), np.array([0.,1.,0.])]:
            vel_ortho = ref - np.dot(ref, Z_hat) * Z_hat
            vel_ortho_mag = np.linalg.norm(vel_ortho)
            if vel_ortho_mag > 1e-6:
                break

    A_hat = vel_ortho / vel_ortho_mag
    C_hat = np.cross(A_hat, Z_hat)
    C_hat = C_hat / np.linalg.norm(C_hat)

    #Verify (all should be < 1e-14 in practice)
    A_hat = np.cross(Z_hat, C_hat)
    A_hat = A_hat / np.linalg.norm(A_hat)
    assert np.abs(np.dot(A_hat, C_hat)) < 1e-6, \
        f"A·C = {np.dot(A_hat, C_hat):.2e} (nan means a norm was zero earlier)"
    assert np.abs(np.dot(A_hat, Z_hat)) < 1e-6, \
        f"A·Z = {np.dot(A_hat, Z_hat):.2e}"
    assert np.abs(np.dot(C_hat, Z_hat)) < 1e-6, \
        f"C·Z = {np.dot(C_hat, Z_hat):.2e}"

    return A_hat, C_hat, Z_hat

def cross_track_distance(point, nadir, C_hat):
    vector_to_point = point - nadir
    distance = np.dot(vector_to_point, C_hat)
    return distance

# ============================================================================
# GEOMETRIC CALCULATIONS
# ============================================================================

def compute_range(spacecraft_pos, surface_pos):
    diff = spacecraft_pos - surface_pos
    range_m = np.linalg.norm(diff)
    return range_m

def compute_incidence_angle(spacecraft_pos, surface_pos, surface_normal):
    look_vector = spacecraft_pos - surface_pos
    look_vector = look_vector / np.linalg.norm(look_vector)
    normal_unit = surface_normal / np.linalg.norm(surface_normal)
    cos_angle = np.abs(np.dot(look_vector, normal_unit))
    cos_angle = np.clip(cos_angle, 0, 1)
    incidence_angle_deg = np.rad2deg(np.arccos(cos_angle))
    return incidence_angle_deg, cos_angle

def compute_surface_normal(corner1, corner2, corner3, corner4):
    edge1 = corner2 - corner1
    edge2 = corner4 - corner1
    normal = np.cross(edge1, edge2)
    center = (corner1 + corner2 + corner3 + corner4) / 4.0
    if np.dot(normal, center) < 0:
        normal = -normal
    return normal

def compute_facet_area(corner1, corner2, corner3, corner4):
    diag1 = corner3 - corner1
    diag2 = corner4 - corner2
    cross = np.cross(diag1, diag2)
    area = 0.5 * np.linalg.norm(cross)
    return area

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_coordinate_transforms():
    print("Testing coordinate transformations...")
    test_cases = [
        (0, 0, 0),           # Equator, prime meridian
        (45, 90, 1000),      # Mid-latitude
        (-45, 180, -2000),   # Southern hemisphere, below datum
        (85, 270, 5000),     # Near pole
    ]
    
    for lat_in, lon_in, h_in in test_cases:
        # Forward transform
        X, Y, Z = geodetic_to_mbfc(lat_in, lon_in, h_in)
        
        # Inverse transform
        lat_out, lon_out, h_out = mbfc_to_geodetic(X, Y, Z)
        
        # Compute errors
        lat_error = abs(lat_out - lat_in)
        lon_error = abs(lon_out - lon_in)
        h_error = abs(h_out - h_in)
        
        print(f"  Input:  ({lat_in:6.2f}, {lon_in:6.2f}, {h_in:7.1f})")
        print(f"  Output: ({lat_out:6.2f}, {lon_out:6.2f}, {h_out:7.1f})")
        print(f"  Errors: ({lat_error:6.4f}°, {lon_error:6.4f}°, {h_error:6.2f} m)")
        
        # Assertions with detailed error messages
        assert lat_error < 1e-6, f"Latitude error {lat_error:.2e} exceeds tolerance"
        assert lon_error < 1e-6, f"Longitude error {lon_error:.2e} exceeds tolerance"
        assert h_error < 0.01, f"Height error {h_error:.2f} m exceeds tolerance"
    
    print("✓ All coordinate transform tests passed!")

def test_local_frame():
    print("\nTesting local coordinate frame...")
    # Create test scenario (spacecraft at 300 km altitude)
    spacecraft = np.array([3696190.0, 0, 0])
    nadir = np.array([3396190.0, 0, 0])
    velocity = np.array([0, 3000, 0])
    
    A_hat, C_hat, Z_hat = compute_local_frame(spacecraft, nadir, velocity)
    
    print(f"  Along-track:  {A_hat}")
    print(f"  Cross-track:  {C_hat}")
    print(f"  Vertical:     {Z_hat}")
    
    # Check unit vectors (magnitude = 1)
    assert abs(np.linalg.norm(A_hat) - 1.0) < 1e-10, "A_hat not unit vector"
    assert abs(np.linalg.norm(C_hat) - 1.0) < 1e-10, "C_hat not unit vector"
    assert abs(np.linalg.norm(Z_hat) - 1.0) < 1e-10, "Z_hat not unit vector"
    
    # Check orthogonality (dot products = 0)
    assert abs(np.dot(A_hat, C_hat)) < 1e-10, "A and C not orthogonal"
    assert abs(np.dot(A_hat, Z_hat)) < 1e-10, "A and Z not orthogonal"
    assert abs(np.dot(C_hat, Z_hat)) < 1e-10, "C and Z not orthogonal"
    
    print("✓ Local frame test passed!")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_coordinate_transforms()
    test_local_frame()