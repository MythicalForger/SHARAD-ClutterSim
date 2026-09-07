from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from sharad import SHARADOrbit

orbit = SHARADOrbit(
    PROJECT_DIR / 'data' / 'sharad' / 's_00571601'
)
print(orbit.get_orbit_info())

radargram = orbit.read_radargram()
print(radargram.shape)

print(orbit.spacecraft_positions[0])   # first trace MBFC position
print(orbit.spacecraft_velocities[0])  # first trace MBFC velocity
print(orbit.nadir_positions[0])