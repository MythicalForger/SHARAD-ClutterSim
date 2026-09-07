"""
Cluttergram assembly for SHARAD clutter simulation.

Converts facet powers into 2D time-delay image.

Key functions:
- Time-delay calculation
- Binning into radargram grid
- Accumulation across facets
- Normalization and output
"""

import numpy as np
from pathlib import Path
import time

from coordinates import geodetic_to_mbfc, compute_local_frame
from facets import generate_facets_at_position, compute_facet_properties_batch
from radar import compute_powers_for_facets, normalize_power_db
import config

# ============================================================================
# TIME-DELAY CALCULATIONS
# ============================================================================

def range_to_time_delay(range_m, speed_of_light=config.SPEED_OF_LIGHT):
    """
    Convert range to two-way time delay.
    
    Formula:
        τ = 2R / c
    
    Factor of 2 because radar signal travels to target and back.
    
    Parameters:
    -----------
    range_m : float or ndarray
        Range in meters
    speed_of_light : float
        Speed of light in m/s (default from config)
    
    Returns:
    --------
    time_delay : float or ndarray
        Two-way time delay in seconds
    
    Example:
        R = 300 km → τ = 2×300000/3e8 = 2.0 ms = 2000 μs
    
    Why two-way:
    ------------
    Signal travels spacecraft → surface → spacecraft
    Total distance = 2 × range
    Time = distance / speed
    """
    
    return 2.0 * range_m / speed_of_light


def time_delay_to_bin(time_delay, time_bin_width=config.SAMPLE_INTERVAL):
    """
    Convert time delay to bin index.
    
    SHARAD samples every 37.5 ns, so each bin represents 37.5 ns.
    
    Parameters:
    -----------
    time_delay : float or ndarray
        Time delay in seconds
    time_bin_width : float
        Width of each time bin in seconds (37.5 ns = 37.5e-9 s)
    
    Returns:
    --------
    bin_index : int or ndarray
        Bin index (integer)
    
    Example:
        τ = 37.5 ns → bin 1
        τ = 75 ns → bin 2
        τ = 50 ns → bin 1 (rounded down)
    
    Why floor (not round)?
    ----------------------
    Standard practice: bin represents range [bin*Δt, (bin+1)*Δt)
    Floor ensures consistent binning.
    
    Alternative: Could use weighted binning (distribute to adjacent bins)
    but simpler approach is adequate.
    """
    
    # Convert time to bin index (floor to integer)
    # np.floor rounds down: 2.7 → 2
    bin_index = np.floor(time_delay / time_bin_width).astype(int)
    
    return bin_index


def compute_relative_time_delay(range_facet, range_nadir, 
                                speed_of_light=config.SPEED_OF_LIGHT):
    """
    Compute time delay relative to nadir.
    
    This is what gets plotted on radargram Y-axis.
    Radargrams show delay AFTER nadir return, not absolute time.
    
    Parameters:
    -----------
    range_facet : float or ndarray
        Range to facet (meters)
    range_nadir : float
        Range to nadir point (meters)
    
    Returns:
    --------
    delta_time : float or ndarray
        Excess time delay beyond nadir (seconds)
    
    Why relative?
    -------------
    Radargrams are triggered at nadir return.
    Time axis shows delay AFTER trigger.
    Nadir return appears at time = 0.
    Surface clutter appears at time > 0.
    Subsurface appears at time > 0 (but from different mechanism).
    
    Formula:
        Δτ = 2(R_facet - R_nadir) / c
    
    Example:
        R_nadir = 300 km
        R_facet = 305 km (5 km farther)
        Δτ = 2×5000/3e8 = 33.3 μs
        → Appears 33.3 μs after nadir in radargram
    """
    
    # Difference in ranges
    delta_range = range_facet - range_nadir
    
    # Convert to time delay (two-way)
    delta_time = 2.0 * delta_range / speed_of_light
    
    return delta_time


# ============================================================================
# CLUTTERGRAM STRUCTURE
# ============================================================================

class Cluttergram:
    """
    Container for cluttergram data and metadata.
    
    Similar to Facet class - groups related data.
    
    Attributes:
    -----------
    data : ndarray, shape (n_positions, n_time_bins)
        Accumulated power (linear units)
    n_positions : int
        Number of along-track positions
    n_time_bins : int
        Number of time delay bins
    time_bin_width : float
        Width of each time bin (seconds)
    max_time_delay : float
        Maximum time delay recorded (seconds)
    spacecraft_positions : ndarray, optional
        Spacecraft positions for each row
    nadir_positions : ndarray, optional
        Nadir positions for each row
    
    Why a class?
    ------------
    - Encapsulates data + metadata
    - Can add methods (normalize, save, plot)
    - Easier to pass around than multiple arrays
    """
    
    def __init__(self, n_positions, n_time_bins=None, max_time_delay=None):
        """
        Initialize empty cluttergram.
        
        Parameters:
        -----------
        n_positions : int
            Number of along-track positions
        n_time_bins : int, optional
            Number of time bins (default from max_time_delay)
        max_time_delay : float, optional
            Maximum time delay in seconds (default from config)
        """
        
        if max_time_delay is None:
            max_time_delay = config.MAX_TIME_DELAY
        
        if n_time_bins is None:
            # Calculate bins from max delay
            n_time_bins = int(max_time_delay / config.SAMPLE_INTERVAL)
        
        self.n_positions = n_positions
        self.n_time_bins = n_time_bins
        self.time_bin_width = config.SAMPLE_INTERVAL
        self.max_time_delay = max_time_delay
        
        # Initialize data array (zeros)
        # dtype=float64 for accumulation (avoid overflow)
        self.data = np.zeros((n_positions, n_time_bins), dtype=np.float64)
        
        # Metadata (optional, filled later)
        self.spacecraft_positions = None
        self.nadir_positions = None
        
    def accumulate_power(self, position_index, time_delays, powers):
        """
        Accumulate power into cluttergram at specific position.
        
        This is the core operation - adding facet powers to correct bins.
        
        Parameters:
        -----------
        position_index : int
            Which along-track position (row in cluttergram)
        time_delays : ndarray
            Time delays for each facet (seconds, relative to nadir)
        powers : ndarray
            Power values for each facet
        
        Algorithm:
        ----------
        1. Convert time delays to bin indices
        2. Filter out-of-bounds bins
        3. Accumulate powers into bins
        
        Why accumulate (not replace)?
        -----------------------------
        Multiple facets may fall in same bin.
        We want total power from all contributing facets.
        This is incoherent summation: P_total = P1 + P2 + P3 + ...
        
        Code notes:
        -----------
        np.add.at: Accumulate at specific indices (handles duplicates correctly)
        Alternative: Loop over bins (slower but clearer)
        """
        
        # Convert time delays to bin indices
        bin_indices = time_delay_to_bin(time_delays, self.time_bin_width)
        
        # Filter: Keep only bins within valid range
        # Bins must be: 0 ≤ bin < n_time_bins
        valid_mask = (bin_indices >= 0) & (bin_indices < self.n_time_bins)
        
        bin_indices_valid = bin_indices[valid_mask]
        powers_valid = powers[valid_mask]
        
        # Accumulate powers into bins
        # np.add.at handles repeated indices correctly
        # data[position_index, bins] += powers
        np.add.at(self.data[position_index, :], bin_indices_valid, powers_valid)
        
    def normalize(self, method='dB', db_range=80.0):
        """
        Normalize cluttergram data to [0, 255] for visualization.
        
        Parameters:
        -----------
        method : str
            'dB', 'linear', or 'sqrt'
        db_range : float
            Dynamic range in dB (for dB method)
        
        Returns:
        --------
        normalized : ndarray, dtype uint8
            Normalized cluttergram
        """
        
        if method == 'dB':
            # Flatten to 1D, normalize, reshape
            data_flat = self.data.ravel()
            
            # Avoid log(0)
            data_safe = data_flat + 1e-30
            
            # Convert to dB
            data_db = 10 * np.log10(data_safe)
            
            # Normalize
            db_max = np.max(data_db)
            db_min = db_max - db_range
            
            data_db_clipped = np.clip(data_db, db_min, db_max)
            normalized_flat = (data_db_clipped - db_min) / db_range
            
            # Scale to [0, 255]
            normalized_flat = (255 * normalized_flat).astype(np.uint8)
            
            # Reshape back
            normalized = normalized_flat.reshape(self.data.shape)
            
        elif method == 'linear':
            d_min = np.min(self.data)
            d_max = np.max(self.data)
            
            if d_max == d_min:
                normalized = np.full_like(self.data, 128, dtype=np.uint8)
            else:
                normalized_float = (self.data - d_min) / (d_max - d_min)
                normalized = (255 * normalized_float).astype(np.uint8)
        
        elif method == 'sqrt':
            data_sqrt = np.sqrt(np.maximum(self.data, 0))
            d_min = np.min(data_sqrt)
            d_max = np.max(data_sqrt)
            
            if d_max == d_min:
                normalized = np.full_like(self.data, 128, dtype=np.uint8)
            else:
                normalized_float = (data_sqrt - d_min) / (d_max - d_min)
                normalized = (255 * normalized_float).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def save(self, filepath):
        """
        Save cluttergram to file.
        
        Saves as NumPy .npz (compressed, includes metadata).
        """
        
        filepath = Path(filepath)
        
        # Build save dict, excluding None metadata to avoid object-array pickle issues
        save_dict = dict(
            data=self.data,
            n_positions=self.n_positions,
            n_time_bins=self.n_time_bins,
            time_bin_width=self.time_bin_width,
            max_time_delay=self.max_time_delay,
        )
        if self.spacecraft_positions is not None:
            save_dict['spacecraft_positions'] = self.spacecraft_positions
        if self.nadir_positions is not None:
            save_dict['nadir_positions'] = self.nadir_positions

        np.savez_compressed(filepath, **save_dict)
        
        print(f"Cluttergram saved: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load cluttergram from file.
        """
        
        filepath = Path(filepath)
        data_loaded = np.load(filepath, allow_pickle=True)
        
        # Create instance
        cluttergram = cls(
            n_positions=int(data_loaded['n_positions']),
            n_time_bins=int(data_loaded['n_time_bins']),
            max_time_delay=float(data_loaded['max_time_delay'])
        )
        
        # Fill data
        cluttergram.data = data_loaded['data']
        
        # Load metadata if present (guard against 0-d object arrays from old saves)
        if 'spacecraft_positions' in data_loaded:
            val = data_loaded['spacecraft_positions']
            cluttergram.spacecraft_positions = None if val.ndim == 0 else val
        if 'nadir_positions' in data_loaded:
            val = data_loaded['nadir_positions']
            cluttergram.nadir_positions = None if val.ndim == 0 else val
        
        return cluttergram


# ============================================================================
# CLUTTERGRAM GENERATION - MAIN FUNCTION
# ============================================================================

def generate_cluttergram(spacecraft_positions, nadir_positions, velocities,
                        dem_interpolator,
                        cross_track_extent=45000,
                        facet_size_cross=30.0,
                        facet_size_along=300.0,
                        max_time_delay=None,
                        apply_angle_cutoff=True,
                        max_incidence_angle=85.0,
                        verbose=True):
    """
    Generate complete cluttergram for orbit.
    
    This is the main entry point - combines all previous sections.
    
    Parameters:
    -----------
    spacecraft_positions : ndarray, shape (n_positions, 3)
        Spacecraft positions in MBFC (meters)
    nadir_positions : ndarray, shape (n_positions, 3)
        Nadir positions in MBFC (meters)
    velocities : ndarray, shape (n_positions, 3)
        Velocity vectors in MBFC (m/s)
    dem_interpolator : BilinearInterpolator
        DEM interpolator (from Section 3)
    cross_track_extent : float
        Half-width of swath (meters)
    facet_size_cross, facet_size_along : float
        Facet dimensions (meters)
    max_time_delay : float, optional
        Maximum time delay to record (seconds)
    apply_angle_cutoff : bool
        Filter high-incidence facets
    max_incidence_angle : float
        Incidence angle cutoff (degrees)
    verbose : bool
        Print progress
    
    Returns:
    --------
    cluttergram : Cluttergram
        Generated cluttergram object
    
    Algorithm:
    ----------
    For each spacecraft position:
        1. Generate facets (Section 4)
        2. Compute powers (Section 5)
        3. Convert ranges to time delays
        4. Accumulate into cluttergram bins
    
    Memory strategy:
    ----------------
    Generate facets on-the-fly (streaming).
    Don't store all facets (would be 18 GB).
    Only keep current position's facets (~1.5 MB).
    Accumulate into cluttergram, then discard facets.
    
    Performance:
    ------------
    For 12,000 positions:
    - ~2 minutes (single-threaded)
    - ~30 seconds (with optimization)
    - Could parallelize across positions
    """
    
    n_positions = len(spacecraft_positions)
    
    if verbose:
        print("="*70)
        print("CLUTTERGRAM GENERATION")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Positions: {n_positions}")
        print(f"  Cross-track: ±{cross_track_extent/1000:.1f} km")
        print(f"  Facet size: {facet_size_cross}m × {facet_size_along}m")
        print(f"  Angle cutoff: {max_incidence_angle}°")
    
    # Initialize cluttergram
    cluttergram = Cluttergram(n_positions, max_time_delay=max_time_delay)
    
    if verbose:
        print(f"\nCluttergram dimensions:")
        print(f"  Along-track: {cluttergram.n_positions}")
        print(f"  Time bins: {cluttergram.n_time_bins}")
        print(f"  Max time delay: {cluttergram.max_time_delay*1e6:.1f} μs")
        print(f"  Memory: {cluttergram.data.nbytes / 1024**2:.1f} MB")
    
    # Progress tracking
    start_time = time.time()
    progress_interval = max(1, n_positions // 20)
    
    # Statistics tracking
    total_facets_generated = 0
    total_facets_accumulated = 0
    
    if verbose:
        print(f"\nGenerating cluttergram...")
    
    # Main loop - process each position
    for i in range(n_positions):
        
        # Get parameters for this position
        sc_pos = spacecraft_positions[i]
        nad_pos = nadir_positions[i]
        vel = velocities[i]
        
        # Compute nadir range (for relative time delays)
        range_nadir = np.linalg.norm(sc_pos - nad_pos)
        
        # Step 1: Generate facets (Section 4)
        facets = generate_facets_at_position(
            sc_pos, nad_pos, vel, dem_interpolator,
            cross_track_extent, facet_size_cross, facet_size_along
        )
        
        total_facets_generated += len(facets)
        
        if len(facets) == 0:
            # No valid facets (outside DEM, all NaN, etc.)
            continue
        
        # Step 2: Extract facet properties (Section 4)
        props = compute_facet_properties_batch(facets)
        
        # Step 3: Compute powers (Section 5)
        from radar import compute_powers_for_facets, apply_incidence_angle_cutoff
        
        powers, ranges, inc_angles, cos_incs = compute_powers_for_facets(props, sc_pos)
        
        # Step 4: Apply incidence angle cutoff (optional)
        if apply_angle_cutoff:
            powers = apply_incidence_angle_cutoff(powers, inc_angles, max_incidence_angle)
        
        # Step 5: Compute time delays (relative to nadir)
        time_delays = compute_relative_time_delay(ranges, range_nadir)
        
        # Step 6: Accumulate into cluttergram
        cluttergram.accumulate_power(i, time_delays, powers)
        
        # Count facets that contributed
        total_facets_accumulated += np.sum(powers > 0)
        
        # Progress update
        if verbose and (i % progress_interval == 0 or i == n_positions - 1):
            elapsed = time.time() - start_time
            progress = (i + 1) / n_positions
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            print(f"  Position {i+1:5d}/{n_positions} ({progress*100:5.1f}%) | "
                  f"Facets: {len(facets):4d} | "
                  f"Elapsed: {elapsed:6.1f}s | "
                  f"ETA: {eta:6.1f}s")
    
    elapsed_total = time.time() - start_time
    
    if verbose:
        print(f"\n" + "="*70)
        print("CLUTTERGRAM GENERATION COMPLETE")
        print("="*70)
        print(f"\nStatistics:")
        print(f"  Total time: {elapsed_total:.1f} seconds")
        print(f"  Time per position: {elapsed_total/n_positions*1000:.1f} ms")
        print(f"  Total facets generated: {total_facets_generated:,}")
        print(f"  Total facets accumulated: {total_facets_accumulated:,}")
        print(f"  Average facets/position: {total_facets_generated/n_positions:.0f}")
        
        # Check cluttergram fill rate
        nonzero_bins = np.sum(cluttergram.data > 0)
        total_bins = cluttergram.data.size
        fill_rate = nonzero_bins / total_bins
        
        print(f"\nCluttergram fill rate:")
        print(f"  Non-zero bins: {nonzero_bins:,} / {total_bins:,}")
        print(f"  Fill rate: {fill_rate*100:.2f}%")
    
    # Store metadata
    cluttergram.spacecraft_positions = spacecraft_positions
    cluttergram.nadir_positions = nadir_positions
    
    return cluttergram


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_cluttergram(cluttergram, normalization='dB', db_range=80.0,
                    figsize=(14, 8), cmap='gray', aspect='auto'):
    """
    Plot cluttergram as 2D image.
    
    Standard radargram-style visualization.
    
    Parameters:
    -----------
    cluttergram : Cluttergram
        Cluttergram object
    normalization : str
        'dB', 'linear', or 'sqrt'
    db_range : float
        Dynamic range for dB normalization
    figsize : tuple
        Figure size
    cmap : str
        Colormap ('gray' is standard for radargrams)
    aspect : str or float
        Aspect ratio ('auto' or numeric)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    
    import matplotlib.pyplot as plt
    
    # Normalize data
    data_normalized = cluttergram.normalize(method=normalization, db_range=db_range)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as image
    # Transpose so time is Y-axis, position is X-axis
    # origin='upper': time increases downward (standard for radargrams)
    im = ax.imshow(data_normalized.T, 
                  cmap=cmap, 
                  aspect=aspect, 
                  origin='upper',
                  interpolation='bilinear')
    
    # Labels
    ax.set_xlabel('Along-track Position', fontsize=12)
    ax.set_ylabel('Time Delay (bin number)', fontsize=12)
    ax.set_title(f'Cluttergram ({normalization} normalization)', 
                fontsize=14, fontweight='bold')
    
    # Convert Y-axis to microseconds
    n_bins = cluttergram.n_time_bins
    max_time_us = cluttergram.max_time_delay * 1e6
    
    # Set Y-tick labels to show time in μs
    y_ticks = ax.get_yticks()
    y_labels = [f'{tick * max_time_us / n_bins:.0f}' for tick in y_ticks]
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Time Delay (μs)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Power (normalized)', fontsize=10)
    
    plt.tight_layout()
    
    return fig, ax