import numpy as np
from scipy import ndimage
from pathlib import Path
from data_readers import MOLAReader
import config

def apply_gaussian_filter(elevation, sigma=2.0):
    print(f"  Applying Gaussian filter (sigma={sigma} pixels)...")
    if np.any(~np.isfinite(elevation)):
        # Create mask of valid data
        valid_mask = np.isfinite(elevation)
        # Replace invalid with mean
        elevation_clean = elevation.copy()
        elevation_clean[~valid_mask] = np.nanmean(elevation)
    else:
        elevation_clean = elevation

    # mode='reflect': extend boundaries by reflection to avoid edge effects
    elevation_filtered = ndimage.gaussian_filter(
        elevation_clean, 
        sigma=sigma, 
        mode='reflect'
    )
    
    if np.any(~np.isfinite(elevation)):
        elevation_filtered[~valid_mask] = np.nan
    diff = elevation - elevation_filtered
    rms_change = np.sqrt(np.nanmean(diff**2))
    print(f"    RMS change: {rms_change:.2f} m")
    return elevation_filtered

def detect_and_mask_outliers(elevation, threshold=3.0):
    median = np.nanmedian(elevation)
    mad = np.nanmedian(np.abs(elevation - median))
    outlier_mask = np.abs(elevation - median) > (threshold * mad)
    n_outliers = np.sum(outlier_mask)
    total_pixels = outlier_mask.size
    outlier_fraction = n_outliers / total_pixels
    print(f"    Found {n_outliers} outliers ({outlier_fraction*100:.3f}%)")
    print(f"    Median elevation: {median:.1f} m")
    print(f"    MAD: {mad:.1f} m")
    elevation_masked = elevation.copy()
    elevation_masked[outlier_mask] = np.nan
    return elevation_masked, outlier_mask

class BilinearInterpolator:
    def __init__(self, elevation, lat_array, lon_array):
        self.elevation = elevation
        self.lat_array = lat_array
        self.lon_array = lon_array
        self.n_lat, self.n_lon = elevation.shape

        self.lat_min = lat_array.min()
        self.lat_max = lat_array.max()
        self.lon_min = lon_array.min()
        self.lon_max = lon_array.max()
        
        self.d_lat = np.abs(lat_array[1] - lat_array[0])
        self.d_lon = lon_array[1] - lon_array[0]
        
        print(f"Interpolator initialized:")
        print(f"  Grid: {self.n_lat} × {self.n_lon}")
        print(f"  Lat range: {self.lat_min:.2f} to {self.lat_max:.2f}°")
        print(f"  Lon range: {self.lon_min:.2f} to {self.lon_max:.2f}°")
        print(f"  Resolution: {self.d_lat:.4f}° × {self.d_lon:.4f}°")
    
    def __call__(self, lat, lon):
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        was_scalar = (lat.size == 1)
        
        #Find grid indices
        lat_idx_float = (self.lat_max - lat) / self.d_lat
        lon_idx_float = (lon - self.lon_min) / self.d_lon
        lat_idx = np.floor(lat_idx_float).astype(int)
        lon_idx = np.floor(lon_idx_float).astype(int)
        u = lat_idx_float - lat_idx
        v = lon_idx_float - lon_idx
        #Bounds checking
        lat_idx = np.clip(lat_idx, 0, self.n_lat - 2)
        lon_idx = np.clip(lon_idx, 0, self.n_lon - 2)
        
        #Get elevation at 4 corners
        h11 = self.elevation[lat_idx, lon_idx]          
        h12 = self.elevation[lat_idx, lon_idx + 1]       
        h21 = self.elevation[lat_idx + 1, lon_idx]      
        h22 = self.elevation[lat_idx + 1, lon_idx + 1]   
        h1 = (1 - u) * h11 + u * h21  
        h2 = (1 - u) * h12 + u * h22  
        h = (1 - v) * h1 + v * h2
        if was_scalar:
            return h[0]
        else:
            return h
    
    def interpolate_grid(self, lat_query, lon_query):
        lon_grid, lat_grid = np.meshgrid(lat_query, lon_query, indexing='ij')
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        
        # Interpolate all points at once (vectorized)
        h_flat = self(lat_flat, lon_flat)
        elevation_grid = h_flat.reshape(lat_grid.shape)
        return elevation_grid

def extract_orbit_corridor(mola_reader, ground_track_lat, ground_track_lon, 
                           cross_track_width=50000):
    print(f"Extracting DEM corridor along orbit...")
    print(f"  Ground track: {len(ground_track_lat)} points")
    print(f"  Cross-track width: ±{cross_track_width/1000:.1f} km")

    lat_min_track = ground_track_lat.min()
    lat_max_track = ground_track_lat.max()
    lon_min_track = ground_track_lon.min()
    lon_max_track = ground_track_lon.max()
    
    # Step 2: Convert cross-track width to degrees (approximate)
    # At equator: 1° ≈ 111 km
    # We use conservative estimate that works everywhere
    METERS_PER_DEGREE = 60000  # Conservative (actually ~111 km at equator)
    margin_deg = cross_track_width / METERS_PER_DEGREE
    
    print(f"  Track lat range: {lat_min_track:.2f} to {lat_max_track:.2f}°")
    print(f"  Track lon range: {lon_min_track:.2f} to {lon_max_track:.2f}°")
    print(f"  Adding margin: ±{margin_deg:.2f}°")
    
    # Step 3: Expand bounding box with margin
    lat_min = lat_min_track - margin_deg
    lat_max = lat_max_track + margin_deg
    lon_min = lon_min_track - margin_deg
    lon_max = lon_max_track + margin_deg
    
    lat_min = np.clip(lat_min, -88, 88)
    lat_max = np.clip(lat_max, -88, 88)
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    
    print(f"  Final extraction range:")
    print(f"    Lat: {lat_min:.2f} to {lat_max:.2f}°")
    print(f"    Lon: {lon_min:.2f} to {lon_max:.2f}°")
    
    # Step 5: Extract from MOLA
    elevation, lat_array, lon_array = mola_reader.read_elevation(
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max)
    )
    
    print(f"  Extracted: {elevation.shape[0]} × {elevation.shape[1]} pixels")
    print(f"  Memory: {elevation.nbytes / 1024**2:.1f} MB")
    
    return elevation, lat_array, lon_array

# ============================================================================
# COMPLETE DEM PROCESSING PIPELINE
# ============================================================================

def process_dem_for_orbit(mola_file, ground_track_lat, ground_track_lon,
                          apply_filter=True, filter_sigma=2.0,
                          detect_outliers=True):
    mola = MOLAReader(mola_file)
    print("\nStep 2: Extracting corridor...")
    elevation, lat_array, lon_array = extract_orbit_corridor(
        mola, ground_track_lat, ground_track_lon
    )
    if detect_outliers:
        print("\nStep 3: Outlier detection...")
        elevation, outlier_mask = detect_and_mask_outliers(elevation)
    else:
        print("\nStep 3: Skipping outlier detection")

    if apply_filter:
        print("\nStep 4: Applying Gaussian filter...")
        elevation = apply_gaussian_filter(elevation, sigma=filter_sigma)
    else:
        print("\nStep 4: Skipping Gaussian filter")
    interpolator = BilinearInterpolator(elevation, lat_array, lon_array)
    return interpolator, elevation, lat_array, lon_array

def create_hillshade(elevation, azimuth=315, altitude=45):
    dy, dx = np.gradient(elevation)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)
    shaded = (
        np.cos(altitude_rad) * np.cos(slope) +
        np.sin(altitude_rad) * np.sin(slope) * 
        np.cos(azimuth_rad - aspect)
    )
    hillshade = 255 * (shaded + 1) / 2
    hillshade = np.clip(hillshade, 0, 255).astype(np.uint8)
    return hillshade