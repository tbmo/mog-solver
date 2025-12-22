# Timespace

A GPU-accelerated N-body gravity simulator using the **Linear-Sparse-Grid** algorithm.

## The Algorithm

Traditional N-body simulations face a fundamental tradeoff: direct particle-to-particle calculations scale O(n²), while grid-based methods (like Particle-Mesh) are fast O(n) but limited by grid resolution. A single high-resolution grid explodes memory usage cubically.

Linear-Sparse-Grid solves this by **stacking multiple low-resolution grids with different SE(3) transforms** (rotations + translations). Each grid is coarse, but because they're offset and rotated relative to each other, particles that fall into the same cell in one grid are separated in others. The force contributions are averaged across all grids.

### Key Insight

Think of two tic-tac-toe boards. Offset one so its gridlines subdivide the other's cells. You have 18 real cells (2 × 3²), but effective sampling resolution of 36 (6²). Each additional grid multiplies effective resolution in each dimension, but only adds linearly to memory.

In 3D with k grids of resolution g:
- **Real cells:** k × g³
- **Effective resolution:** (k × g)³
- **Memory savings:** k²

With 360 grids × 16³: actual memory = 1.47M cells, effective resolution = 5760³ = 191 billion cells. That's **130,000x memory savings**.

The transforms are generated using a 6D Halton sequence mapped through Hopf fibration (for uniform SO(3) coverage) and scaled translations, ensuring deterministic quasi-random coverage of SE(3) space.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  For each grid g ∈ [0, n_grids):                                │
│    1. Transform particle positions by R_g (rotation) + t_g      │
│    2. Scatter mass to grid cells (atomic add)                   │
│    3. Solve for gravitational force field:                      │
│       - FFT mode: FFT → multiply by Green's function → IFFT     │
│       - Convolution mode: direct spatial convolution            │
│    4. Sample forces at transformed particle positions           │
│    5. Rotate forces back to world space                         │
└─────────────────────────────────────────────────────────────────┘
│  Average forces across all grids                                │
│  Update velocities and positions (leapfrog integration)         │
└─────────────────────────────────────────────────────────────────┘
```

# Find the staggered grid paper
# Find everything that has cited it
# Make sure nothing better than my algorithm

# Generalized to the material point method


## Requirements

- Python 3.8+
- OpenGL 4.6 capable GPU
- Dependencies: `numpy`, `moderngl`, `moderngl-window`, `pyyaml`

## Usage

```bash
pip install numpy moderngl moderngl-window pyyaml
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| Mouse drag | Rotate camera |
| Scroll | Zoom in/out |
| Space | Pause/resume |
| R | Reset particles (Perlin noise distribution) |
| T | Reset with two colliding galaxies |
| 1/2/3 | Particle clustering presets |
| G | Toggle grid debug visualization |
| H | Toggle particle rendering |
| C | Toggle between FFT and direct convolution solvers |
| [ / ] | Decrease/increase timescale |
| \ | Reverse time |
| Q | Quit |

## Configuration

Edit `config.yaml`:

```yaml
n_grids: 360        # Number of SE(3)-transformed grids
grid_size: 16       # Resolution per grid (16³)
world_size: 1000.0  # Simulation domain size
num_particles: 5e5  # Particle count
G: 1.0              # Gravitational constant
dt: 0.00001         # Time step
solver: fft         # "fft" or "convolution"
damping: 0.99       # Velocity damping per step
```

## Complexity

| Approach | Memory | Time per step |
|----------|--------|---------------|
| Direct N-body | O(n) | O(n²) |
| Single PM grid | O(g³) | O(g³ log g) |
| **Linear-Sparse-Grid** | O(k·g³) | O(n·k + k·g³ log g) |

Where n = particles, g = grid size, k = number of grids.
