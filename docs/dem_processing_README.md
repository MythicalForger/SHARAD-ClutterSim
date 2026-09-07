# Section 3: DEM Processing - Code Explanation

## Overview

This section takes raw MOLA elevation data and transforms it into a smooth, interpolatable surface ready for facet generation.

**Input:** Gridded elevation data (463m resolution, with artifacts)

**Output:** Continuous interpolation function (query any lat/lon)

---

## Key Concepts

### 1. Why Filter the DEM?

**MOLA striping artifacts:**
- Caused by interpolation between orbital tracks
- Systematic N-S stripes at ~500m spacing
- Creates false topography → false clutter

**Solution: Gaussian filter**
- Low-pass filter (removes high frequencies)
- Preserves long-wavelength topography (>2 km)
- Removes short-wavelength artifacts (<1 km)

### 2. Why Bilinear Interpolation?

**Problem:**
- Facets at 30m spacing
- MOLA at 463m spacing
- Need elevation between grid points

**Bilinear interpolation:**
- Linear in both directions (lat and lon)
- Smooth (C⁰ continuous)
- Fast (no transcendental functions)
- Good enough for our purposes

**Alternative: Cubic**
- Smoother (C¹ continuous)
- Slower (4× more points, more computation)
- Can overshoot (create artificial peaks/valleys)
- Not worth complexity for this application

### 3. Why Extract Corridors?

**Memory consideration:**
- Full MOLA: 530 MB
- Single orbit corridor: 16 MB
- Typical simulation: 10-20 orbits → 160-320 MB vs 530 MB

**Speed consideration:**
- Less data → faster filtering
- Less data → faster interpolation
- Can keep multiple orbits in memory

---

## Code Patterns Explained

### Pattern 1: Gaussian Filtering

```python
from scipy import ndimage

# Simple Gaussian filter
filtered = ndimage.gaussian_filter(data, sigma=2.0, mode='reflect')
```

**What each part does:**

- `sigma=2.0`: Width of Gaussian kernel (in pixels)
  - Larger σ → more smoothing
  - σ=2 is empirically good for MOLA

- `mode='reflect'`: Edge handling
  - Mirrors data at boundaries
  - Prevents edge darkening
  - Alternatives: 'constant', 'nearest', 'wrap'

**How it works internally:**
1. Create Gaussian kernel G(x,y) = exp(-(x²+y²)/2σ²)
2. Convolve with image: out[i,j] = Σ G(x,y) × in[i+x, j+y]
3. Normalize so sum of kernel = 1

### Pattern 2: NumPy Indexing

```python
# Get 4 corners for interpolation
h11 = elevation[lat_idx, lon_idx]
h12 = elevation[lat_idx, lon_idx + 1]
h21 = elevation[lat_idx + 1, lon_idx]
h22 = elevation[lat_idx + 1, lon_idx + 1]
```

**Why this works:**
- `lat_idx` and `lon_idx` can be arrays
- NumPy broadcasts the indexing
- Gets all required corners in one operation
- Much faster than Python loop

**Example:**
```python
lat_idx = np.array([10, 20, 30])
lon_idx = np.array([5, 15, 25])

h11 = elevation[lat_idx, lon_idx]
# Returns: [elevation[10,5], elevation[20,15], elevation[30,25]]
```

### Pattern 3: Bilinear Interpolation

```python
# Normalized position within cell
u = lat_idx_float - lat_idx  # ∈ [0, 1]
v = lon_idx_float - lon_idx  # ∈ [0, 1]

# Interpolate
h1 = (1 - u) * h11 + u * h21  # Along one edge
h2 = (1 - u) * h12 + u * h22  # Along other edge
h = (1 - v) * h1 + v * h2     # Between edges
```

**Geometric interpretation:**