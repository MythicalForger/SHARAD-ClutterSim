import numpy as np
import struct
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import rasterio


def parse_pds_label(label_file):
    params = {}
    
    with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    patterns = {
        'LINES': r'LINES\s*=\s*(\d+)',
        'LINE_SAMPLES': r'LINE_SAMPLES\s*=\s*(\d+)',
        'SAMPLE_BITS': r'SAMPLE_BITS\s*=\s*(\d+)',
        'SAMPLE_TYPE': r'SAMPLE_TYPE\s*=\s*["\']?(\w+)["\']?',
        'OFFSET': r'^\s*OFFSET\s*=\s*([-+]?\d+\.?\d*)',
        'SCALING_FACTOR': r'SCALING_FACTOR\s*=\s*([-+]?\d+\.?\d*)',
        'BYTES': r'BYTES\s*=\s*(\d+)',
        'RECORD_BYTES': r'RECORD_BYTES\s*=\s*(\d+)',
        'MAXIMUM_LATITUDE':      r'MAXIMUM_LATITUDE\s*=\s*([-+]?\d+\.?\d*)',
        'MINIMUM_LATITUDE':      r'MINIMUM_LATITUDE\s*=\s*([-+]?\d+\.?\d*)',
        'WESTERNMOST_LONGITUDE': r'WESTERNMOST_LONGITUDE\s*=\s*([-+]?\d+\.?\d*)',
        'EASTERNMOST_LONGITUDE': r'EASTERNMOST_LONGITUDE\s*=\s*([-+]?\d+\.?\d*)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1)
            if key in ['LINES', 'LINE_SAMPLES', 'SAMPLE_BITS', 'BYTES', 'RECORD_BYTES']:
                params[key] = int(value)
            elif key in ['OFFSET', 'SCALING_FACTOR',
                        'MAXIMUM_LATITUDE', 'MINIMUM_LATITUDE',
                        'WESTERNMOST_LONGITUDE', 'EASTERNMOST_LONGITUDE']:
                params[key] = float(value)
            else:
                params[key] = value
    
    # Set defaults if not found
    params.setdefault('OFFSET', 0.0)
    params.setdefault('SCALING_FACTOR', 1.0)
    
    return params

class MOLAReader:
    def __init__(self, img_file, lbl_file=None):
        self.img_file = Path(img_file)
        
        if lbl_file is None:
            # Try .lbl.txt first, then .lbl
            if (self.img_file.parent / f"{self.img_file.stem}.lbl.txt").exists():
                self.lbl_file = self.img_file.parent / f"{self.img_file.stem}.lbl.txt"
            elif self.img_file.with_suffix('.lbl').exists():
                self.lbl_file = self.img_file.with_suffix('.lbl')
            else:
                raise FileNotFoundError(f"Could not find label file for {self.img_file}")
        else:
            self.lbl_file = Path(lbl_file)

        print(f"Reading MOLA DEM:")
        print(f"  IMG: {self.img_file.name}")
        print(f"  LBL: {self.lbl_file.name}")
        
        # Parse label
        self.params = parse_pds_label(self.lbl_file)
        self.n_lat = self.params.get('LINES', 23040)  # Default for MEGDR 128ppd
        self.n_lon = self.params.get('LINE_SAMPLES', 11520)
        self.sample_bits = self.params.get('SAMPLE_BITS', 16)
        
        print(f"\n  Dimensions: {self.n_lat} × {self.n_lon}")
        print(f"  Sample bits: {self.sample_bits}")
        print(f"  Offset: {self.params['OFFSET']}")
        print(f"  Scaling: {self.params['SCALING_FACTOR']}")
        
    def read_elevation(self, lat_range=None, lon_range=None):      
        # Determine data type
        sample_type = self.params.get('SAMPLE_TYPE', 'LSB_INTEGER').upper()
        if self.sample_bits == 16:
            dtype = np.dtype('>i2') if 'MSB' in sample_type else np.dtype('<i2')
        elif self.sample_bits == 32:
            if 'REAL' in sample_type:
                dtype = np.dtype('<f4')  
            else:
                dtype = np.dtype('>i4') if 'MSB' in sample_type else np.dtype('<i4')
        else:
            raise ValueError(f"Unsupported sample bits: {self.sample_bits}")
        
        # Generate full coordinate arrays first
        max_lat = self.params.get('MAXIMUM_LATITUDE', 88.0)
        min_lat = self.params.get('MINIMUM_LATITUDE', -88.0)
        west_lon = self.params.get('WESTERNMOST_LONGITUDE', 0.0)
        east_lon = self.params.get('EASTERNMOST_LONGITUDE', 360.0)
        lat_full = np.linspace(max_lat, min_lat, self.n_lat)
        lon_full = np.linspace(west_lon, east_lon, self.n_lon, endpoint=False)
        
        # Determine subset indices
        if lat_range is not None:
            lat_min, lat_max = lat_range
            lat_idx_min = np.argmin(np.abs(lat_full - lat_max))  # Note: max first
            lat_idx_max = np.argmin(np.abs(lat_full - lat_min))
            lat_indices = slice(lat_idx_min, lat_idx_max + 1)
            lat_array = lat_full[lat_indices]
        else:
            lat_indices = slice(None)
            lat_array = lat_full
        
        if lon_range is not None:
            lon_min, lon_max = lon_range
            lon_idx_min = np.argmin(np.abs(lon_full - lon_min))
            lon_idx_max = np.argmin(np.abs(lon_full - lon_max))
            lon_indices = slice(lon_idx_min, lon_idx_max + 1)
            lon_array = lon_full[lon_indices]
        else:
            lon_indices = slice(None)
            lon_array = lon_full
        
        #Read DEM
        if lat_range is None and lon_range is None:
            print("  Reading entire DEM (this may take a moment)...")
            data = np.fromfile(self.img_file, dtype=dtype)
            elevation_raw = data.reshape(self.n_lat, self.n_lon)
        else:
            print(f"  Reading subset: lat[{lat_indices}], lon[{lon_indices}]")
            bytes_per_sample = self.sample_bits // 8
            record_bytes = self.n_lon * bytes_per_sample
            elevation_raw = np.zeros((len(lat_array), self.n_lon), dtype=dtype)
            with open(self.img_file, 'rb') as f:
                for i, lat_idx in enumerate(range(lat_indices.start or 0, 
                                                   lat_indices.stop or self.n_lat)):
                    f.seek(lat_idx * record_bytes)
                    row_data = np.fromfile(f, dtype=dtype, count=self.n_lon)
                    elevation_raw[i, :] = row_data
            
            # Extract longitude subset
            elevation_raw = elevation_raw[:, lon_indices]
        
        offset = self.params['OFFSET']
        scaling = self.params['SCALING_FACTOR'] 
        elevation = offset + elevation_raw.astype(np.float32) * scaling
        print(f"  Elevation range: {elevation.min():.0f} to {elevation.max():.0f} m")
        return elevation, lat_array, lon_array

class SHARADRadargram:
    def __init__(self, img_file, lbl_file=None, xml_file=None):
        self.img_file = Path(img_file)

        if lbl_file is None:
            if (self.img_file.parent / f"{self.img_file.stem}.lbl.txt").exists():
                self.lbl_file = self.img_file.parent / f"{self.img_file.stem}.lbl.txt"
            else:
                # Try with .rgram.lbl.txt
                base_name = str(self.img_file.name).replace('.img', '')
                lbl_txt = self.img_file.parent / f"{base_name}.lbl.txt"
                if lbl_txt.exists():
                    self.lbl_file = lbl_txt
                else:
                    raise FileNotFoundError(f"Could not find label file for {self.img_file}")
        else:
            self.lbl_file = Path(lbl_file)
        
        if xml_file is None:
            xml_path = self.img_file.with_suffix('.xml')
            if xml_path.exists():
                self.xml_file = xml_path
            else:
                self.xml_file = None
                print("⚠️  XML file not found (geometry will be limited)")
        else:
            self.xml_file = Path(xml_file)
        
        print(f"Reading SHARAD radargram:")
        print(f"  IMG: {self.img_file.name}")
        print(f"  LBL: {self.lbl_file.name}")
        if self.xml_file:
            print(f"  XML: {self.xml_file.name}")
        
        self.params = parse_pds_label(self.lbl_file)        
        # Extract dimensions
        self.n_traces = self.params.get('LINE_SAMPLES', 0)  # Along-track
        self.n_samples = self.params.get('LINES', 0)        # Time samples        
        print(f"\n  Traces (along-track): {self.n_traces}")
        print(f"  Samples (time): {self.n_samples}")        
        self.geometry = None
        if self.xml_file:
            self._parse_geometry()
    
    def _parse_geometry(self):
        """Parse XML file for spacecraft geometry"""
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
            
            # This is simplified - actual XML structure may vary
            # You'll need to inspect the XML to find the right tags
            
            print("\n⚠️  XML parsing needs customization for your file format")
            print("    Open the XML file to see structure")
            
            self.geometry = {
                'parsed': False,
                'message': 'XML structure needs manual inspection'
            }
            
        except Exception as e:
            print(f"⚠️  Could not parse XML: {e}")
            self.geometry = None
    
    def read_radargram(self):
        print("\nReading radargram...")
        # Determine data type
        sample_bits = self.params.get('SAMPLE_BITS', 32)
        if sample_bits == 32:
            dtype = np.float32
        elif sample_bits == 16:
            dtype = np.int16
        else:
            dtype = np.float32
            print(f"⚠️  Unknown sample bits ({sample_bits}), assuming float32")
        
        # Read binary data
        data = np.fromfile(self.img_file, dtype=dtype)
        
        # Reshape to 2D
        try:
            radargram = data.reshape(self.n_samples, self.n_traces)
            print(f"  Shape: {radargram.shape}")
            print(f"  Data range: {radargram.min():.2e} to {radargram.max():.2e}")
        except ValueError as e:
            print(f"⚠️  Reshape failed: {e}")
            print(f"    Data size: {data.size}")
            print(f"    Expected: {self.n_samples} × {self.n_traces} = {self.n_samples * self.n_traces}")
            return None
        
        return radargram

#Loading Functions
def load_mola_subset(mola_dir, lat_range, lon_range):
    mola_dir = Path(mola_dir)
    img_file = mola_dir / 'megt88n000hb.img'
    
    mola = MOLAReader(img_file)
    elevation, lats, lons = mola.read_elevation(lat_range, lon_range)
    
    return elevation, lats, lons

def load_sharad_orbit(sharad_dir, orbit_number):
    sharad_dir = Path(sharad_dir)
    orbit_str = f"{int(orbit_number):08d}"  # e.g., '00571601'
    
    # Find files with this orbit number
    img_file = list(sharad_dir.glob(f"*{orbit_str}*rgram.img"))
    
    if not img_file:
        raise FileNotFoundError(f"No radargram found for orbit {orbit_number}")
    
    sharad = SHARADRadargram(img_file[0])
    radargram = sharad.read_radargram()
    
    return sharad, radargram