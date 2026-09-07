import numpy as np
def spacecraft_to_local(spacecraft_mbfc, nadir_mbfc, e_hat, n_hat, r_hat):
    offset = spacecraft_mbfc - nadir_mbfc
    return np.array([
        np.dot(offset, e_hat),
        np.dot(offset, n_hat),
        np.dot(offset, r_hat),
    ])

def compute_range_and_incidence(facet_centers_local, facet_normals_local, spacecraft_local):
    diff = spacecraft_local[None, :] - facet_centers_local  # (n, 3)
    ranges = np.linalg.norm(diff, axis=1)

    look_unit = diff / ranges[:, None]
    normals_unit = facet_normals_local / np.linalg.norm(facet_normals_local, axis=1, keepdims=True)

    cos_incidence = np.abs(np.sum(look_unit * normals_unit, axis=1))
    cos_incidence = np.clip(cos_incidence, 0.0, 1.0)
    incidence_deg = np.degrees(np.arccos(cos_incidence))

    return ranges, incidence_deg, cos_incidence


def compute_received_power(facet_area, range_m, cos_incidence):
    return (facet_area ** 4) * (cos_incidence ** 4) / (range_m ** 4)


def apply_incidence_angle_cutoff(powers, incidence_deg, max_angle=85.0):
    powers_filtered = powers.copy()
    powers_filtered[incidence_deg > max_angle] = 0.0
    return powers_filtered


def normalize_power_db(powers, db_range=80.0):
    powers_safe = powers + 1e-30
    powers_db = 10 * np.log10(powers_safe)

    p_db_max = np.max(powers_db)
    p_db_min = p_db_max - db_range

    powers_db_clipped = np.clip(powers_db, p_db_min, p_db_max)
    normalized = (powers_db_clipped - p_db_min) / db_range
    return (255 * normalized).astype(np.uint8)  