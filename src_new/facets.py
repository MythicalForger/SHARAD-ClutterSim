#Assumption 1: For surface normal and area calculation, we are just using the facet corner points to get a general gradient of the facet
import numpy as np
MARS_RADIUS = 3396190.0

def generate_facet_row(nadir_lat, nadir_lon, heading_deg, dem,
                        cross_track_extent=45000.0,
                        facet_size_cross=30.0,
                        facet_size_along=300.0):
    """
    Generate one cross-track row of facets at a single along-track position.
    
    Returns
    -------
    dict of arrays, one row per facet (facets with any NaN corner are dropped):
        'center_lat', 'center_lon', 'center_elev' : (n,)
        'corners_lat', 'corners_lon', 'corners_elev' : (n, 4)
        'cross_track_dist' : (n,)  signed distance from nadir, meters
        'area' : (n,)  facet area, m^2 (from real corner geometry)
        'normal_local' : (n, 3)  normal vector in local ENU frame (east, north, up)            
    """
    half_cross = facet_size_cross / 2.0
    half_along = facet_size_along / 2.0

    cross_track_dist = np.arange(
        -cross_track_extent,
        cross_track_extent + facet_size_cross,
        facet_size_cross,
    )
    n_facets = len(cross_track_dist)

    corner_offsets_local = np.array([
        [-half_cross, -half_along],
        [+half_cross, -half_along],
        [+half_cross, +half_along],
        [-half_cross, +half_along],
    ])

    #Array Broadcasting: cross_track_dist is made into (n,1) matrix and corner_offsets_local is made into (1,4) matrix, then broadcasting is done
    d_cross = cross_track_dist[:, None] + corner_offsets_local[None, :, 0]  # final size of both d_cross and d_along is (n,4)
    d_along = np.zeros_like(d_cross) + corner_offsets_local[None, :, 1]  

    # Rotate (cross, along) -> (east, north) using heading, then convert
    # to lat/lon offsets via flat local approximation (validated: ~0.3m
    # error over a 5km offset, negligible vs 30m facet size).
    heading_rad = np.radians(heading_deg)
    fwd_east, fwd_north = np.sin(heading_rad), np.cos(heading_rad)
    right_east, right_north = np.sin(heading_rad + np.pi / 2), np.cos(heading_rad + np.pi / 2)
    d_east = d_cross * right_east + d_along * fwd_east
    d_north = d_cross * right_north + d_along * fwd_north
    m_per_deg_lat = np.pi * MARS_RADIUS / 180.0
    m_per_deg_lon = np.pi * MARS_RADIUS * np.cos(np.radians(nadir_lat)) / 180.0
    corners_lat = nadir_lat + d_north / m_per_deg_lat   # (n, 4)
    corners_lon = nadir_lon + d_east / m_per_deg_lon    # (n, 4)

    flat_lat = corners_lat.ravel()
    flat_lon = corners_lon.ravel()
    flat_elev = dem.get_elevation(flat_lat, flat_lon)
    corners_elev = flat_elev.reshape(n_facets, 4)

    # Drop facets where any corner is NaN (outside DEM / nodata)
    valid = np.all(np.isfinite(corners_elev), axis=1)

    corners_lat = corners_lat[valid]
    corners_lon = corners_lon[valid]
    corners_elev = corners_elev[valid]
    cross_track_dist_valid = cross_track_dist[valid]

    if corners_lat.shape[0] == 0:
        return _empty_facet_row()

    # ------------------------------------------------------------------
    # Facet geometry in a local east-north-up frame, using elevation as
    # the "up" displacement. This is an approximation (true MBFC corner
    # positions need coordinates.py's geodetic_to_mbfc), but is enough to
    # get a normal vector and area at facet scale, consistent with how
    # elevation varies corner-to-corner.
    # ------------------------------------------------------------------
    d_east_v = d_east[valid]
    d_north_v = d_north[valid]
    corners_xyz_local = np.stack([d_east_v, d_north_v, corners_elev], axis=-1)  # (n,4,3)

    # normal via cross product of diagonals
    diag1 = corners_xyz_local[:, 2, :] - corners_xyz_local[:, 0, :]  # corner3 - corner1
    diag2 = corners_xyz_local[:, 3, :] - corners_xyz_local[:, 1, :]  # corner4 - corner2
    normal = np.cross(diag1, diag2)
    area = 0.5 * np.linalg.norm(normal, axis=1)

    # ensure normal points "up" (positive local-up component)
    flip = normal[:, 2] < 0
    normal[flip] *= -1

    normal_unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    center_lat = corners_lat.mean(axis=1)
    center_lon = corners_lon.mean(axis=1)
    center_elev = corners_elev.mean(axis=1)

    return {
        'center_lat': center_lat,
        'center_lon': center_lon,
        'center_elev': center_elev,
        'corners_lat': corners_lat,
        'corners_lon': corners_lon,
        'corners_elev': corners_elev,
        'cross_track_dist': cross_track_dist_valid,
        'area': area,
        'normal_local': normal_unit,
    }


def _empty_facet_row():
    return {
        'center_lat': np.array([]), 'center_lon': np.array([]), 'center_elev': np.array([]),
        'corners_lat': np.zeros((0, 4)), 'corners_lon': np.zeros((0, 4)), 'corners_elev': np.zeros((0, 4)),
        'cross_track_dist': np.array([]), 'area': np.array([]), 'normal_local': np.zeros((0, 3)),
    }