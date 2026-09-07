Function inventory, all 7 modules

1. dem.py ✅ done

MarsDEM.__init__(tif_path)
MarsDEM.get_elevation(lat, lon) — bilinear, vectorized
MarsDEM.__call__ — alias

2. sharad.py

SHARADOrbit.__init__(base_path) — opens .img/.lbl/.xml trio
_parse_geometry_xml() — sc position/velocity, nadir lat/lon/radius arrays
read_radargram() — via planetaryimage or raw np.fromfile
get_orbit_info() — orbit number, record count, lat/lon/altitude ranges

3. coordinates.py

geodetic_to_mbfc(lat, lon, height) — via pyproj transformer
mbfc_to_geodetic(x, y, z) — via pyproj transformer (inverse)
compute_local_frame(sc_pos, nadir_pos, velocity) — hand-written Gram-Schmidt
(drop cross_track_distance, compute_range, compute_incidence_angle, compute_surface_normal, compute_facet_area — these are one-liners, inline them into facets.py/radar.py directly rather than importing trivial wrappers)

4. facets.py

generate_facets_at_position(sc_pos, nadir_pos, velocity, dem, cross_track_extent, facet_size_cross, facet_size_along) — vectorized across the whole cross-track row at once (batch DEM query instead of per-corner loop)
compute_facet_geometry(corners) — normal + area from corner arrays, vectorized

5. radar.py

compute_received_power(facet_area, range_m, cos_incidence) — the A⁴cos⁴θ/R⁴ formula
apply_incidence_angle_cutoff(powers, angles, max_angle)
normalize_power_db(powers, db_range)

6. cluttergram.py

Cluttergram.__init__(n_positions, max_time_delay)
Cluttergram.accumulate_power(position_index, time_delays, powers)
Cluttergram.normalize(method)
Cluttergram.save/load(filepath)
range_to_time_delay, compute_relative_time_delay — small, could fold into the class or keep standalone

7. pipeline.py

run_orbit_simulation(sharad_base_path, dem_path) — wires 2→3→4→5→6 for one orbit
plot_comparison(cluttergram, radargram) — sim vs. real side-by-side



