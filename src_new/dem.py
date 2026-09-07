import numpy as np
import rasterio
from scipy.ndimage import map_coordinates
MARS_RADIUS = 3396190.0 

class MarsDEM:
    def __init__(self, tif_path):
        self.src = rasterio.open(tif_path)
        self.transform = self.src.transform
        self.inv_transform = ~self.transform
        self.width = self.src.width
        self.height = self.src.height
        self.nodata = self.src.nodata
        self.band = self.src.read(1)

    def close(self):
        self.src.close()

    def lonlat_to_pixel(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lon = ((lon + 180) % 360) - 180 #Wrap from SHARAD 0 to 360deg coords to -180 to 180deg coords for mola dem
        # since src.crs gives equirectangular projection, so we treat it as cylindrical projection (as in a cylinder rolled out as a rectangle and the top polar regions ignored ofc)
        x = MARS_RADIUS * np.radians(lon)
        y = MARS_RADIUS * np.radians(lat)
        col, row = self.inv_transform * (x, y)
        return row, col

    def get_elevation(self, lat, lon):
        scalar_input = np.isscalar(lat) or np.ndim(lat) == 0
 
        row, col = self.lonlat_to_pixel(lat, lon)
        row = np.atleast_1d(row)
        col = np.atleast_1d(col)
        row0 = np.floor(row).astype(int)
        col0 = np.floor(col).astype(int)
        valid = (
            (row0 >= 0) & (row0 + 1 < self.height) &
            (col0 >= 0) & (col0 + 1 < self.width)
        ) #bounds cheking
 
        #order 0:nearest edge, 1:billinear, 2:quadratic, 3:cubic
        interp = map_coordinates(
            self.band, [row, col], order=1, mode='nearest'
        ).astype(np.float64)

        if self.nodata is not None:
            r0 = np.clip(row0, 0, self.height - 1)
            r1 = np.clip(row0 + 1, 0, self.height - 1)
            c0 = np.clip(col0, 0, self.width - 1)
            c1 = np.clip(col0 + 1, 0, self.width - 1)
            corner_nodata = (
                (self.band[r0, c0] == self.nodata) |
                (self.band[r0, c1] == self.nodata) |
                (self.band[r1, c0] == self.nodata) |
                (self.band[r1, c1] == self.nodata)
            )
            valid = valid & ~corner_nodata
 
        result = np.full(row.shape, np.nan, dtype=np.float64)
        result[valid] = interp[valid]
 
        if scalar_input:
            return float(result[0])
        return result

    def __call__(self, lat, lon):
        return self.get_elevation(lat, lon)