"""
SHARAD data reader for RDR (Reduced Data Record) products.

Reads:
- Radargrams (.img files)
- Geometry data (ancillary .lbl and .xml files)
- Ephemeris (spacecraft position, velocity)
"""

import numpy as np
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

class SHARADReader:
    """
    Reader for SHARAD RDR data products.
    
    SHARAD RDR format:
    - Binary .img file (radargram data)
    - PDS .lbl file (metadata)
    - XML file (geometry, ephemeris)
    
    Data structure:
    - Each record = one along-track position
    - Each record has samples in time (range bins)
    - Plus auxiliary data (geometry, ephemeris)
    """
    
    def __init__(self, base_path):
        """
        Initialize SHARAD reader.
        
        Parameters:
        -----------
        base_path : str or Path
            Base path to SHARAD files (without extension)
            Example: 'data/sharad/s_00571601_rgram'
            Will look for:
                - s_00571601_rgram.img
                - s_00571601_rgram.lbl.txt
                - s_00571601_rgram.xml
        """
        
        self.base_path = Path(base_path)
        
        # Construct file paths
        self.img_file = self.base_path.with_suffix('.img')
        self.lbl_file = Path(str(self.base_path) + '.lbl.txt')
        self.xml_file = self.base_path.with_suffix('.xml')
        
        # Verify files exist
        if not self.img_file.exists():
            raise FileNotFoundError(f"IMG file not found: {self.img_file}")
        if not self.xml_file.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_file}")
        
        print(f"SHARAD Reader initialized:")
        print(f"  IMG: {self.img_file.name}")
        print(f"  XML: {self.xml_file.name}")
        
        # Parse metadata
        self._parse_xml()
    
    def _parse_xml(self):
        """
        Parse XML file to extract geometry and ephemeris.
        
        XML contains for each record:
        - Spacecraft position (X, Y, Z in MBFC)
        - Spacecraft velocity (Vx, Vy, Vz)
        - Surface intersection point (lat, lon, radius)
        - Timing information
        """
        
        print("\nParsing XML geometry file...")
        
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Find geometry records
        # XML structure varies by SHARAD version, need to handle carefully
        
        # Try to find ancillary_data section
        ancillary = root.find('.//ancillary_data')
        if ancillary is None:
            # Try alternate structure
            ancillary = root.find('.//Ancillary_Data')
        
        if ancillary is None:
            raise ValueError("Cannot find ancillary_data in XML")
        
        # Extract arrays
        # These are stored as text with whitespace-separated values
        
        def parse_array(element_name):
            """Helper to parse array from XML element."""
            elem = ancillary.find(element_name)
            if elem is None:
                # Try with different case
                elem = ancillary.find(element_name.lower())
            if elem is None:
                elem = ancillary.find(element_name.upper())
            
            if elem is None:
                return None
            
            # Parse text as space-separated numbers
            text = elem.text.strip()
            values = [float(x) for x in text.split()]
            return np.array(values)
        
        # Spacecraft position (meters in MBFC)
        self.sc_pos_x = parse_array('SPACECRAFT_POSITION_X')
        self.sc_pos_y = parse_array('SPACECRAFT_POSITION_Y')
        self.sc_pos_z = parse_array('SPACECRAFT_POSITION_Z')
        
        # Spacecraft velocity (m/s in MBFC)
        self.sc_vel_x = parse_array('SPACECRAFT_VELOCITY_X')
        self.sc_vel_y = parse_array('SPACECRAFT_VELOCITY_Y')
        self.sc_vel_z = parse_array('SPACECRAFT_VELOCITY_Z')
        
        # Surface intersection (nadir point)
        self.surface_lat = parse_array('SUB_SC_PLANETOCENTRIC_LATITUDE')
        self.surface_lon = parse_array('SUB_SC_EAST_LONGITUDE')
        self.surface_radius = parse_array('MARS_RADIUS')  # meters from center
        
        # Validate we got data
        if self.sc_pos_x is None:
            raise ValueError("Failed to parse spacecraft position from XML")
        
        self.n_records = len(self.sc_pos_x)
        
        print(f"  Found {self.n_records} geometry records")
        print(f"  Spacecraft position range:")
        print(f"    X: {self.sc_pos_x.min()/1e6:.3f} to {self.sc_pos_x.max()/1e6:.3f} Mm")
        print(f"    Y: {self.sc_pos_y.min()/1e6:.3f} to {self.sc_pos_y.max()/1e6:.3f} Mm")
        print(f"    Z: {self.sc_pos_z.min()/1e6:.3f} to {self.sc_pos_z.max()/1e6:.3f} Mm")
        print(f"  Ground track:")
        print(f"    Lat: {self.surface_lat.min():.2f}° to {self.surface_lat.max():.2f}°")
        print(f"    Lon: {self.surface_lon.min():.2f}° to {self.surface_lon.max():.2f}°")
    
    def get_spacecraft_positions(self):
        """
        Get spacecraft positions as (n_records, 3) array.
        
        Returns:
        --------
        positions : ndarray, shape (n_records, 3)
            Spacecraft positions in MBFC (X, Y, Z) in meters
        """
        
        return np.column_stack([self.sc_pos_x, self.sc_pos_y, self.sc_pos_z])
    
    def get_spacecraft_velocities(self):
        """
        Get spacecraft velocities as (n_records, 3) array.
        
        Returns:
        --------
        velocities : ndarray, shape (n_records, 3)
            Velocity vectors in MBFC (Vx, Vy, Vz) in m/s
        """
        
        return np.column_stack([self.sc_vel_x, self.sc_vel_y, self.sc_vel_z])
    
    def get_ground_track(self):
        """
        Get ground track coordinates.
        
        Returns:
        --------
        latitudes : ndarray
            Planetocentric latitudes (degrees)
        longitudes : ndarray
            East longitudes (degrees)
        """
        
        return self.surface_lat, self.surface_lon
    
    def get_nadir_positions(self):
        """
        Get nadir positions in MBFC.
        
        Converts (lat, lon, radius) to MBFC coordinates.
        
        Returns:
        --------
        nadir_positions : ndarray, shape (n_records, 3)
            Nadir positions in MBFC
        """
        
        from coordinates import geodetic_to_mbfc
        
        nadir_positions = []
        
        for lat, lon, radius in zip(self.surface_lat, self.surface_lon, self.surface_radius):
            # Convert planetocentric to geodetic (approximately equal for Mars)
            # SHARAD uses planetocentric, we need geodetic for our functions
            # For Mars (low flattening), difference is small
            
            # Height above ellipsoid
            from config import MARS_RADIUS_EQUATOR
            height = radius - MARS_RADIUS_EQUATOR
            
            X, Y, Z = geodetic_to_mbfc(lat, lon, height)
            nadir_positions.append([X, Y, Z])
        
        return np.array(nadir_positions)
    
    def read_radargram(self):
        """
        Read radargram data from .img file.
        
        SHARAD RDR format:
        - 32-bit float values
        - Shape: (n_records, n_samples)
        - n_samples typically 3600
        
        Returns:
        --------
        radargram : ndarray, shape (n_records, n_samples)
            Radargram data (power values)
        """
        
        print("\nReading radargram data...")
        
        # SHARAD RDR files are raw binary
        # Need to determine dimensions
        
        # Get file size
        file_size = self.img_file.stat().st_size
        
        # SHARAD RDR: 32-bit floats
        bytes_per_sample = 4
        
        # Typical SHARAD: 3600 samples per record
        # Try to infer from file size
        samples_per_record = 3600
        
        expected_size = self.n_records * samples_per_record * bytes_per_sample
        
        if file_size != expected_size:
            print(f"  ⚠️  File size mismatch: {file_size} vs expected {expected_size}")
            print(f"     Trying to infer dimensions...")
            
            # Recalculate samples
            samples_per_record = file_size // (self.n_records * bytes_per_sample)
            print(f"     Inferred samples per record: {samples_per_record}")
        
        # Read binary data
        data = np.fromfile(self.img_file, dtype='>f4')  # Big-endian 32-bit float
        
        # Reshape
        try:
            radargram = data.reshape(self.n_records, samples_per_record)
        except ValueError:
            print(f"  ⚠️  Cannot reshape to ({self.n_records}, {samples_per_record})")
            # Try transposed
            samples_per_record = len(data) // self.n_records
            radargram = data.reshape(self.n_records, samples_per_record)
        
        print(f"  Radargram shape: {radargram.shape}")
        print(f"  Value range: {radargram.min():.2e} to {radargram.max():.2e}")
        
        return radargram
    
    def get_orbit_info(self):
        """
        Get summary information about orbit.
        
        Returns dict with:
        - orbit_number
        - n_records
        - lat_range
        - lon_range
        - altitude_range
        """
        
        # Extract orbit number from filename
        # Format: s_XXXXXXXX_rgram.img where XXXXXXXX is orbit number
        orbit_number = int(self.img_file.stem.split('_')[1])
        
        # Compute altitude
        sc_pos = self.get_spacecraft_positions()
        nadir_pos = self.get_nadir_positions()
        
        altitudes = np.linalg.norm(sc_pos - nadir_pos, axis=1)
        
        info = {
            'orbit_number': orbit_number,
            'n_records': self.n_records,
            'lat_range': (self.surface_lat.min(), self.surface_lat.max()),
            'lon_range': (self.surface_lon.min(), self.surface_lon.max()),
            'altitude_range': (altitudes.min(), altitudes.max()),
            'altitude_mean': altitudes.mean(),
        }
        
        return info


def visualize_orbit_geometry(reader):
    """
    Visualize orbit geometry from SHARAD reader.
    
    Creates plots showing:
    - Ground track
    - Altitude profile
    - Velocity profile
    """
    
    import matplotlib.pyplot as plt
    
    # Get data
    lats, lons = reader.get_ground_track()
    sc_pos = reader.get_spacecraft_positions()
    nadir_pos = reader.get_nadir_positions()
    velocities = reader.get_spacecraft_velocities()
    
    # Compute altitudes
    altitudes = np.linalg.norm(sc_pos - nadir_pos, axis=1)
    
    # Compute velocity magnitudes
    vel_mags = np.linalg.norm(velocities, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Ground track
    ax1 = axes[0, 0]
    scatter = ax1.scatter(lons, lats, c=altitudes/1000, cmap='viridis', s=1)
    ax1.set_xlabel('Longitude (°E)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title('Ground Track', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Altitude (km)')
    
    # Altitude profile
    ax2 = axes[0, 1]
    ax2.plot(altitudes/1000, 'b-', linewidth=1)
    ax2.set_xlabel('Record Number', fontsize=12)
    ax2.set_ylabel('Altitude (km)', fontsize=12)
    ax2.set_title('Altitude Profile', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(altitudes.mean()/1000, color='r', linestyle='--', 
                label=f'Mean: {altitudes.mean()/1000:.1f} km')
    ax2.legend()
    
    # Velocity profile
    ax3 = axes[1, 0]
    ax3.plot(vel_mags/1000, 'g-', linewidth=1)
    ax3.set_xlabel('Record Number', fontsize=12)
    ax3.set_ylabel('Velocity (km/s)', fontsize=12)
    ax3.set_title('Velocity Magnitude', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(vel_mags.mean()/1000, color='r', linestyle='--',
                label=f'Mean: {vel_mags.mean()/1000:.2f} km/s')
    ax3.legend()
    
    # 3D trajectory
    ax4 = axes[1, 1]
    ax4.plot(sc_pos[:, 0]/1e6, sc_pos[:, 1]/1e6, 'b-', linewidth=1, label='Spacecraft')
    ax4.plot(nadir_pos[:, 0]/1e6, nadir_pos[:, 1]/1e6, 'r-', linewidth=1, label='Nadir')
    ax4.set_xlabel('X (Mm)', fontsize=12)
    ax4.set_ylabel('Y (Mm)', fontsize=12)
    ax4.set_title('Trajectory (X-Y Plane)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axis('equal')
    
    plt.tight_layout()
    
    return fig