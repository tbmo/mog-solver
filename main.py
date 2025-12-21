"""
Linear-Sparse-Grid N-Body Solver

Achieves high effective resolution by stacking multiple low-resolution grids
with diagonal spatial offsets, scaling linearly O(n) rather than quadratically O(n²).

Supports both 2D and 3D simulations.

For 3D: Each grid is processed independently to avoid spectral leakage.
Grids are stored in separate 3D textures and processed sequentially for FFT.
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import yaml

import pickle
import hashlib


class HierarchicalSE3Sampler:
    """
    Jointly optimizes rotation + translation for grid sampling.

    The configuration space is SO(3) × T³ where T³ is the 3-torus of
    translations mod voxel_size. We want each (rotation, offset) pair
    to create a maximally different grid tessellation of space.
    """

    def __init__(self, grid_size, world_size, n_grids, seed=42, cache_dir=".se3_cache"):
        self.grid_size = grid_size
        self.world_size = world_size
        self.voxel_size = world_size / grid_size
        self.n_grids = n_grids
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._configs = []
        self._max_computed = 0

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_key = self._compute_cache_key()
        self.cache_path = self.cache_dir / f"{self.cache_key}.pkl"

        if self._load_cache():
            print(f"  Loaded {len(self._configs)} configs from cache")

        self.w_rot, self.w_trans = self._compute_weights()

    def _compute_cache_key(self):
        """Generate a unique hash for these parameters."""
        params = f"gs{self.grid_size}_ws{self.world_size}_ng{self.n_grids}_s{self.seed}"
        return hashlib.md5(params.encode()).hexdigest()[:12]

    def _load_cache(self):
        """Try to load configs from cache file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    self._configs = data["configs"]
                    self._max_computed = data["max_computed"]
                    return True
            except Exception as e:
                print(f"  Cache load failed: {e}")
        return False

    def _save_cache(self):
        """Save current configs to cache file."""
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "configs": self._configs,
                        "max_computed": self._max_computed,
                        "params": {
                            "grid_size": self.grid_size,
                            "world_size": self.world_size,
                            "n_grids": self.n_grids,
                            "seed": self.seed,
                        },
                    },
                    f,
                )
            print(f"  Saved {len(self._configs)} configs to cache")
        except Exception as e:
            print(f"  Cache save failed: {e}")

    def _compute_weights(self):
        """
        Auto-tune rotation vs translation weights.

        Key insight: A rotation of angle θ moves points at distance r by ~rθ.
        A translation moves all points equally.

        For a grid of size G in world of size W:
        - Typical point distance from grid origin: ~W/2
        - Voxel size: V = W/G
        - Max useful rotation effect: ~(W/2) × π = πW/2
        - Max useful translation effect: V × √3/2 (diagonal of voxel)

        We want both to contribute comparably to cell boundary movement.
        """
        V = self.voxel_size
        W = self.world_size
        G = self.grid_size

        char_radius = W / (2 * np.sqrt(3))

        max_rot_displacement = char_radius * np.pi

        max_trans_displacement = V * np.sqrt(3) / 2

        ratio = max_trans_displacement / max_rot_displacement

        grid_factor = min(1.0, 60 / self.n_grids)

        coarseness = 32 / G

        base_rot = 0.6
        base_trans = 0.4

        coarse_adjust = 0.15 * (coarseness - 1)

        count_adjust = 0.1 * (1 - grid_factor)

        w_rot = np.clip(base_rot - coarse_adjust + count_adjust, 0.3, 0.8)
        w_trans = 1 - w_rot

        print(f"SE(3) Sampler Auto-Tuning:")
        print(f"  voxel_size: {V:.2f}")
        print(f"  char_radius: {char_radius:.2f}")
        print(f"  max_rot_displacement: {max_rot_displacement:.2f}")
        print(f"  max_trans_displacement: {max_trans_displacement:.2f}")
        print(f"  coarseness factor: {coarseness:.2f} (32/G)")
        print(f"  grid count factor: {grid_factor:.2f}")
        print(f"  → weights: rotation={w_rot:.2f}, translation={w_trans:.2f}")

        return w_rot, w_trans

    def quat_to_matrix(self, q):
        """Convert unit quaternion [w,x,y,z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype="f4",
        )

    def config_distance(self, c1, c2):
        """
        Distance between two (rotation, offset) configurations.
        Uses auto-tuned weights for rotation vs translation.
        """
        q1, off1 = c1
        q2, off2 = c2

        dot = np.abs(np.dot(q1, q2))
        rot_dist = 2 * np.arccos(np.clip(dot, 0, 1)) / np.pi

        diff = off1 - off2
        diff = diff - self.voxel_size * np.round(diff / self.voxel_size)
        trans_dist = np.linalg.norm(diff) / (self.voxel_size * np.sqrt(3) / 2)

        return self.w_rot * rot_dist + self.w_trans * trans_dist

    def min_distance_to_set(self, config):
        """Minimum distance from config to all existing configurations."""
        if len(self._configs) == 0:
            return 1.0
        return min(self.config_distance(config, c) for c in self._configs)

    def get_icosahedral_quaternions(self):
        """
        Generate all 60 icosahedral rotation quaternions.
        Uses the binary icosahedral group (120 quaternions on S³),
        then takes one representative from each ±q pair.
        """
        phi = (1 + np.sqrt(5)) / 2
        quats = []

        for val in [1, -1]:
            quats.append([val, 0, 0, 0])
            quats.append([0, val, 0, 0])
            quats.append([0, 0, val, 0])
            quats.append([0, 0, 0, val])

        for s0 in [1, -1]:
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    for s3 in [1, -1]:
                        quats.append([0.5 * s0, 0.5 * s1, 0.5 * s2, 0.5 * s3])

        coords = [0, 1, phi, 1 / phi]

        even_perms = [
            (0, 1, 2, 3),
            (0, 2, 3, 1),
            (0, 3, 1, 2),
            (1, 2, 0, 3),
            (1, 3, 2, 0),
            (1, 0, 3, 2),
            (2, 0, 1, 3),
            (2, 3, 0, 1),
            (2, 1, 3, 0),
            (3, 1, 0, 2),
            (3, 0, 2, 1),
            (3, 2, 1, 0),
        ]

        for perm in even_perms:
            vals = [coords[perm[0]], coords[perm[1]], coords[perm[2]], coords[perm[3]]]
            signs_list = [[1]]
            for v in vals[1:]:
                if v == 0:
                    signs_list.append([1])
                else:
                    signs_list.append([1, -1])

            for s1 in signs_list[1]:
                for s2 in signs_list[2]:
                    for s3 in signs_list[3]:
                        q = [
                            0.5 * vals[0],
                            0.5 * s1 * vals[1],
                            0.5 * s2 * vals[2],
                            0.5 * s3 * vals[3],
                        ]
                        quats.append(q)

        quats = [np.array(q, dtype="f4") for q in quats]
        quats = [q / np.linalg.norm(q) for q in quats]

        unique = []
        for q in quats:
            q = self._canonicalize_quat(q)
            is_dup = any(np.abs(np.abs(np.dot(q, u)) - 1) < 1e-4 for u in unique)
            if not is_dup:
                unique.append(q)

        print(f"  Generated {len(unique)} unique icosahedral rotations")
        return unique[:60]

    def _canonicalize_quat(self, q):
        """Ensure quaternion is in positive hemisphere."""
        q = q / np.linalg.norm(q)
        for i in range(4):
            if abs(q[i]) > 1e-6:
                return np.array(-q if q[i] < 0 else q, dtype="f4")
        return np.array(q, dtype="f4")

    def _random_config(self):
        """Generate random (quaternion, offset) configuration."""
        q = self.rng.standard_normal(4).astype("f4")
        q = self._canonicalize_quat(q)
        off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
        return (q, off)

    def _optimize_offset_for_rotation(self, q, n_candidates=200):
        """Find best offset for a given rotation quaternion."""
        best_off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
        best_dist = self.min_distance_to_set((q, best_off))

        for _ in range(n_candidates):
            off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
            d = self.min_distance_to_set((q, off))
            if d > best_dist:
                best_dist = d
                best_off = off

        return best_off, best_dist

    def compute_hierarchical(self, n, verbose=False):
        """
        Compute n (rotation, offset) configurations hierarchically.

        Strategy:
        1. First 60: icosahedral rotations with optimized offsets
        2. 61-120: same rotations, new optimized offsets (subdivides!)
        3. Beyond: full SE(3) farthest-point sampling
        """
        if n <= self._max_computed:
            return self._configs[:n]

        icosa_quats = self.get_icosahedral_quaternions()

        while self._max_computed < min(n, len(icosa_quats) * 4):
            cycle = self._max_computed // 60
            idx = self._max_computed % 60

            q = icosa_quats[idx]
            off, dist = self._optimize_offset_for_rotation(
                q, n_candidates=150 + cycle * 50
            )

            self._configs.append((q, off))
            self._max_computed += 1

            if verbose and self._max_computed % 20 == 0:
                print(
                    f"  Config {self._max_computed}: cycle={cycle}, min_dist={dist:.4f}"
                )

        if n > self._max_computed:
            n_candidates = max(500, (n - self._max_computed) * 10)
            candidates = [self._random_config() for _ in range(n_candidates)]

            while self._max_computed < n:
                best_dist = -1
                best_config = None
                best_idx = -1

                for i, config in enumerate(candidates):
                    d = self.min_distance_to_set(config)
                    if d > best_dist:
                        best_dist = d
                        best_config = config
                        best_idx = i

                self._configs.append(best_config)
                self._max_computed += 1

                if verbose and self._max_computed % 20 == 0:
                    print(
                        f"  Config {self._max_computed}: full SE(3), min_dist={best_dist:.4f}"
                    )

                candidates.pop(best_idx)
                candidates = [
                    c
                    for c in candidates
                    if self.config_distance(c, best_config) > best_dist * 0.3
                ]

                if len(candidates) < (n - self._max_computed) * 3:
                    candidates.extend([self._random_config() for _ in range(300)])

        self._save_cache()

        return self._configs[:n]

    def get_transforms(self, n=None, verbose=False):
        """
        Get (rotation_matrix, offset) pairs for grid transforms.

        Returns:
            rotations: (n, 3, 3) array of rotation matrices
            offsets: (n, 3) array of translation offsets
        """
        if n is None:
            n = self.n_grids

        configs = self.compute_hierarchical(n, verbose)

        rotations = np.zeros((n, 3, 3), dtype="f4")
        offsets = np.zeros((n, 3), dtype="f4")

        for i, (q, off) in enumerate(configs):
            rotations[i] = self.quat_to_matrix(q)
            offsets[i] = off

        return rotations, offsets

    def analyze_coverage(self, n=None):
        """Analyze the quality of the configuration set."""
        if n is None:
            n = len(self._configs)
        configs = self._configs[:n]

        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(self.config_distance(configs[i], configs[j]))

        dists = np.array(dists)

        print(f"\nCoverage Analysis (n={n}):")
        print(f"  Min pairwise distance: {dists.min():.4f}")
        print(f"  Mean pairwise distance: {dists.mean():.4f}")
        print(f"  Max pairwise distance: {dists.max():.4f}")
        print(f"  Std pairwise distance: {dists.std():.4f}")

        test_configs = [self._random_config() for _ in range(1000)]
        max_gap = max(self.min_distance_to_set(c) for c in test_configs)
        print(f"  Estimated covering radius: {max_gap:.4f}")

        return dists


def set_uniform(prog, name, value):
    """Set uniform only if it exists (wasn't optimized out)."""
    if name in prog:
        prog[name] = value


def create_perspective_projection(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, (2 * far * near) / (near - far), 0],
        ],
        dtype="f4",
    )


def create_look_at(eye, target, up):
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    view = np.identity(4, dtype="f4")
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[0, 3] = -np.dot(x_axis, eye)
    view[1, 3] = -np.dot(y_axis, eye)
    view[2, 3] = -np.dot(z_axis, eye)

    return view.T


class Simulation(mglw.WindowConfig):
    gl_version = (4, 6)
    title = "Linear-Sparse-Grid N-Body"
    window_size = (1024, 1024)
    aspect_ratio = 1.0
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = self.load_config()

        self.dim = self.cfg.get("dimensions", 2)
        if self.dim not in [2, 3]:
            raise NotImplementedError("Only 2D and 3D supported.")

        self.timescale = 1.0
        self.paused = False

        self.n_grids = int(self.cfg.get("n_grids", 16))
        self.grid_size = int(self.cfg["grid_size"])
        self.world_size = float(self.cfg["world_size"])
        self.voxel_size = self.world_size / self.grid_size
        self.num_particles = int(float(self.cfg["num_particles"]))
        self.G = self.cfg["G"]
        self.dt = self.cfg["dt"]

        self.offsets, self.rotations = self.compute_transforms()

        self.init_buffers()
        self.init_textures()
        self.init_shaders()
        self.init_particles()
        self.init_render()

        self.cam_rot_x = 0.0
        self.cam_rot_y = 0.0
        self.cam_dist = 3.0
        self.mouse_pressed = False
        self.show_grid_debug = False

        effective_res = self.grid_size * self.n_grids
        total_cells = self.n_grids * (self.grid_size**self.dim)
        equiv_cells = effective_res**self.dim

        print(f"Linear-Sparse-Grid Config ({self.dim}D):")
        print(f"  world_size: {self.world_size}")
        print(f"  num_particles: {self.num_particles:,}")
        print(f"  n_grids: {self.n_grids}")
        print(f"  grid_size: {self.grid_size}^{self.dim}")
        print(f"  Total cells: {total_cells:,}")
        print(
            f"  Equivalent single grid: {effective_res}^{self.dim} = {equiv_cells:,} cells"
        )
        print(f"  Memory savings: {equiv_cells / total_cells:.1f}x")

    def get_icosahedral_rotations(self):
        """
        Returns all 60 proper rotations of an icosahedron.
        """
        rotations = []
        phi = (1 + np.sqrt(5)) / 2

        icosa_verts = np.array(
            [
                [0, 1, phi],
                [0, 1, -phi],
                [0, -1, phi],
                [0, -1, -phi],
                [1, phi, 0],
                [1, -phi, 0],
                [-1, phi, 0],
                [-1, -phi, 0],
                [phi, 0, 1],
                [-phi, 0, 1],
                [phi, 0, -1],
                [-phi, 0, -1],
            ],
            dtype="f4",
        )
        icosa_verts /= np.linalg.norm(icosa_verts[0])

        def rotation_matrix(axis, angle):
            """Rodrigues' rotation formula"""
            axis = axis / np.linalg.norm(axis)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        def matrix_key(m):
            """Hash a rotation matrix for deduplication"""
            return tuple(np.round(m.flatten() * 1000).astype(int))

        seen = set()

        def add_rotation(mat):
            key = matrix_key(mat)
            if key not in seen:
                seen.add(key)
                rotations.append(mat.astype("f4"))

        add_rotation(np.eye(3))

        for v in icosa_verts:
            for k in [1, 2, 3, 4]:
                angle = k * 2 * np.pi / 5
                add_rotation(rotation_matrix(v, angle))

        faces = [
            (0, 2, 8),
            (0, 8, 4),
            (0, 4, 6),
            (0, 6, 9),
            (0, 9, 2),
            (3, 1, 10),
            (3, 10, 5),
            (3, 5, 7),
            (3, 7, 11),
            (3, 11, 1),
            (2, 9, 7),
            (2, 7, 5),
            (2, 5, 8),
            (8, 5, 10),
            (8, 10, 4),
            (4, 10, 1),
            (4, 1, 6),
            (6, 1, 11),
            (6, 11, 9),
            (9, 11, 7),
        ]
        for f in faces:
            center = icosa_verts[f[0]] + icosa_verts[f[1]] + icosa_verts[f[2]]
            center /= np.linalg.norm(center)
            for k in [1, 2]:
                angle = k * 2 * np.pi / 3
                add_rotation(rotation_matrix(center, angle))

        edges = set()
        for f in faces:
            edges.add((min(f[0], f[1]), max(f[0], f[1])))
            edges.add((min(f[1], f[2]), max(f[1], f[2])))
            edges.add((min(f[2], f[0]), max(f[2], f[0])))

        for e in edges:
            mid = icosa_verts[e[0]] + icosa_verts[e[1]]
            mid /= np.linalg.norm(mid)
            add_rotation(rotation_matrix(mid, np.pi))

        print(f"Generated {len(rotations)} icosahedral rotations")

        assert len(rotations) == 60, f"Expected 60 rotations, got {len(rotations)}"

        return rotations

    def get_dihedral_rotations(self):
        """Returns all 8 rotations/reflections of a square as 2x2 matrices (D4 group)."""
        rotations = []

        for k in range(4):
            angle = k * np.pi / 2
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype="f4")
            rotations.append(rot)

        reflect = np.array([[1, 0], [0, -1]], dtype="f4")
        for k in range(4):
            angle = k * np.pi / 2
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype="f4")
            rotations.append(rot @ reflect)

        return rotations

    def compute_transforms(self):
        offsets = np.zeros((self.n_grids, 4), dtype="f4")
        rotations = np.zeros((self.n_grids, 12), dtype="f4")

        if self.dim == 2:
            dihedral = self.get_dihedral_rotations()
            diag_dirs = np.array(
                [[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype="f4"
            ) / np.sqrt(2)
            for i in range(self.n_grids):
                frac = i / self.n_grids
                offsets[i, 0:2] = frac * self.voxel_size * diag_dirs[i % 4]
                rot = dihedral[i % 8]
                rotations[i, 0:2] = rot[:, 0]
                rotations[i, 4:6] = rot[:, 1]
        else:
            sampler = HierarchicalSE3Sampler(
                grid_size=self.grid_size,
                world_size=self.world_size,
                n_grids=self.n_grids,
                seed=12345,
            )
            rot_mats, offs = sampler.get_transforms(verbose=True)
            sampler.analyze_coverage()

            for i in range(self.n_grids):
                offsets[i, 0:3] = offs[i]
                rotations[i, 0:3] = rot_mats[i, :, 0]
                rotations[i, 4:7] = rot_mats[i, :, 1]
                rotations[i, 8:11] = rot_mats[i, :, 2]

        return offsets, rotations

    def load_config(self):
        cfg_path = Path(__file__).parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f)
        return {
            "n_grids": 16,
            "grid_size": 32,
            "world_size": 100.0,
            "num_particles": 4096,
            "G": 1.0,
            "dt": 0.016,
            "particle_mass": 1.0,
            "dimensions": 2,
        }

    def init_buffers(self):
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())
        self.rotation_buf = self.ctx.buffer(self.rotations.tobytes())

        cells_per_grid = self.grid_size**self.dim
        total_cells = self.n_grids * cells_per_grid
        self.mass_buf = self.ctx.buffer(reserve=total_cells * 4)

    def init_textures(self):
        gs = self.grid_size
        ng = self.n_grids

        if self.dim == 2:
            size = (gs, gs, ng)

            def create_tex(channels):
                return self.ctx.texture3d(size, channels, dtype="f4")

            self.complex_tex_a = create_tex(2)
            self.complex_tex_b = create_tex(2)

            self.grad_comp_x = create_tex(2)
            self.grad_comp_y = create_tex(2)
            self.grad_comp_z = None

            self.gradient_tex = create_tex(4)

        else:
            size = (gs, gs, gs * ng)

            def create_tex(channels):
                return self.ctx.texture3d(size, channels, dtype="f4")

            self.complex_tex_a = create_tex(2)
            self.complex_tex_b = create_tex(2)

            self.grad_comp_x = create_tex(2)
            self.grad_comp_y = create_tex(2)
            self.grad_comp_z = create_tex(2)

            self.gradient_tex = create_tex(4)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.wnd.mouse_states.left:
            self.cam_rot_x -= dx * 0.5
            self.cam_rot_y += dy * 0.5

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.cam_dist -= y_offset * 0.2
        self.cam_dist = max(0.1, self.cam_dist)

    def init_shaders(self):
        shader_dir = Path(__file__).parent / "shaders"

        def load(name):
            src = (shader_dir / name).read_text()
            header = f"#version 460\n#define DIM {self.dim}\n#define N_GRIDS {self.n_grids}\n#define GRID_SIZE {self.grid_size}\n"
            src = src.replace("#version 460", "")
            return header + src

        self.prog_mass = self.ctx.compute_shader(load("mass.glsl"))
        self.prog_complex = self.ctx.compute_shader(load("complex.glsl"))
        self.prog_fft = self.ctx.compute_shader(load("fft.glsl"))
        self.prog_fft_batched = self.ctx.compute_shader(load("fft_batched.glsl"))
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))
        self.prog_fourier = self.ctx.compute_shader(load("fourier.glsl"))
        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))
        self.prog_init_particles = self.ctx.compute_shader(load("init_particles.glsl"))

    def init_particles(self):
        """Initialize particles using GPU Perlin noise for non-uniform distribution."""
        self.init_particles_gpu(
            noise_scale=4.0,
            density_contrast=2.0,
            seed=None,
        )

    def init_particles_gpu(
        self, noise_scale=4.0, density_contrast=2.0, seed=None, spawn_buffer=None
    ):
        """Initialize particles on GPU with Perlin noise density field.

        Args:
            noise_scale: Controls noise frequency. Higher = more/smaller clusters.
            density_contrast: Controls clustering strength. 1.0 = uniform, higher = more clustered.
            seed: Random seed for reproducibility. None = random seed.
            spawn_buffer: Fraction of world_size to keep clear from edges (0.0-0.5). None = use config.
        """
        if seed is None:
            seed = np.random.randint(0, 2**31)
        if spawn_buffer is None:
            spawn_buffer = self.cfg.get("spawn_buffer", 0.1)

        n_particles = self.num_particles

        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)

        self.prog_init_particles["numParticles"] = n_particles
        self.prog_init_particles["worldSize"] = float(self.world_size)
        self.prog_init_particles["noiseScale"] = float(noise_scale)
        self.prog_init_particles["densityContrast"] = float(density_contrast)
        self.prog_init_particles["seed"] = int(seed)
        self.prog_init_particles["baseMass"] = float(self.cfg["particle_mass"])
        self.prog_init_particles["spawnBuffer"] = float(spawn_buffer)

        self.prog_init_particles.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        print(
            f"GPU init: {n_particles:,} particles, scale={noise_scale}, contrast={density_contrast}, buffer={spawn_buffer:.0%}"
        )

    def init_particles_cpu(self):
        """Initialize particles in a simple centered cluster (CPU fallback)."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        center = self.world_size / 2.0
        spread = self.world_size * 0.2

        pos[:, : self.dim] = center + np.random.uniform(
            -spread, spread, (self.num_particles, self.dim)
        )
        pos[:, 3] = self.cfg["particle_mass"]

        self.pos_buf.write(pos.tobytes())
        self.vel_buf.write(vel.tobytes())

    def init_particles2(self):
        """Initialize two colliding disk galaxies."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        center = self.world_size / 2.0
        n_per_galaxy = self.num_particles // 2

        for i, (offset, v_bulk) in enumerate(
            [
                (
                    np.array([-0.2, 0.0, 0.0]),
                    np.array([0.3, 0.1, 0.0]),
                ),
                (
                    np.array([0.2, 0.0, 0.0]),
                    np.array([-0.3, -0.1, 0.0]),
                ),
            ]
        ):
            start = i * n_per_galaxy
            end = start + n_per_galaxy

            r = np.random.exponential(scale=0.08, size=n_per_galaxy) * self.world_size
            theta = np.random.uniform(0, 2 * np.pi, n_per_galaxy)
            phi = np.arccos(np.random.uniform(-1, 1, n_per_galaxy))

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            pos[start:end, 0] = center + offset[0] * self.world_size + x
            pos[start:end, 1] = center + offset[1] * self.world_size + y
            pos[start:end, 2] = center + offset[2] * self.world_size + z
            pos[start:end, 3] = self.cfg["particle_mass"]

            spin_axis = (
                np.array([0.3, 0.7, 0.5]) if i == 0 else np.array([-0.5, 0.2, 0.8])
            )
            spin_axis = spin_axis / np.linalg.norm(spin_axis)

            v_circ = 0.3 * np.sqrt(r / self.world_size + 0.01)
            pos_vec = np.stack([x, y, z], axis=1)
            v_dir = np.cross(spin_axis, pos_vec)
            v_dir_norm = np.linalg.norm(v_dir, axis=1, keepdims=True) + 1e-10
            v_dir = v_dir / v_dir_norm

            vel[start:end, 0] = v_circ * v_dir[:, 0] + v_bulk[0]
            vel[start:end, 1] = v_circ * v_dir[:, 1] + v_bulk[1]
            vel[start:end, 2] = v_circ * v_dir[:, 2] + v_bulk[2]

        self.pos_buf.write(pos.tobytes())
        self.vel_buf.write(vel.tobytes())

    def init_render(self):
        self.render_prog = self.ctx.program(
            vertex_shader="""
                #version 460
                layout(location = 0) in vec4 in_pos;

                uniform float world_size;
                uniform mat4 m_proj;
                uniform mat4 m_view;

                void main() {
                    vec3 p = (in_pos.xyz / world_size) * 2.0 - 1.0;
                    gl_Position = m_proj * m_view * vec4(p, 1.0);
                    float dist = gl_Position.w;
                    gl_PointSize = 600.0 / (dist + 0.1);
                }
                """,
            fragment_shader="""
                #version 460
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
                }
                """,
        )
        self.render_prog["world_size"] = self.world_size
        self.vao = self.ctx.vertex_array(
            self.render_prog, [(self.pos_buf, "4f", "in_pos")]
        )

        self.grid_debug_prog = self.ctx.program(
            vertex_shader="""
                #version 460
                layout(location = 0) in vec3 in_pos;
                layout(location = 1) in vec3 in_color;

                uniform float world_size;
                uniform mat4 m_proj;
                uniform mat4 m_view;

                out vec3 v_color;

                void main() {
                    vec3 p = (in_pos / world_size) * 2.0 - 1.0;
                    gl_Position = m_proj * m_view * vec4(p, 1.0);
                    v_color = in_color;
                }
                """,
            fragment_shader="""
                #version 460
                in vec3 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 0.6);
                }
                """,
        )
        self.grid_debug_prog["world_size"] = self.world_size
        self.init_grid_debug_geometry()

    def init_grid_debug_geometry(self):
        """Create single-cell geometry for each grid to visualize offsets and rotations."""

        def hsv_to_rgb(h, s, v):
            i = int(h * 6)
            f = h * 6 - i
            p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
            i = i % 6
            if i == 0:
                return (v, t, p)
            if i == 1:
                return (q, v, p)
            if i == 2:
                return (p, v, t)
            if i == 3:
                return (p, q, v)
            if i == 4:
                return (t, p, v)
            return (v, p, q)

        vs = self.voxel_size
        center = self.world_size / 2.0

        positions = []
        colors = []

        for grid_idx in range(self.n_grids):
            hue = (grid_idx * 0.618033988749895) % 1.0
            r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
            color = (r, g, b)
            offset = self.offsets[grid_idx][: self.dim]

            rot_data = self.rotations[grid_idx]

            if self.dim == 2:
                rot = np.array(
                    [[rot_data[0], rot_data[4]], [rot_data[1], rot_data[5]]], dtype="f4"
                )

                corners = np.array(
                    [
                        [0, 0],
                        [vs, 0],
                        [vs, vs],
                        [0, vs],
                    ],
                    dtype="f4",
                )

                cell_center = np.array([vs / 2, vs / 2], dtype="f4")
                rotated_corners = []
                for c in corners:
                    local = c - cell_center
                    rotated = rot @ local
                    world = (
                        rotated
                        + cell_center
                        + np.array([center + offset[0], center + offset[1]], dtype="f4")
                    )
                    rotated_corners.append(world)

                positions.extend(
                    [
                        [rotated_corners[0][0], rotated_corners[0][1], 0],
                        [rotated_corners[1][0], rotated_corners[1][1], 0],
                        [rotated_corners[1][0], rotated_corners[1][1], 0],
                        [rotated_corners[2][0], rotated_corners[2][1], 0],
                        [rotated_corners[2][0], rotated_corners[2][1], 0],
                        [rotated_corners[3][0], rotated_corners[3][1], 0],
                        [rotated_corners[3][0], rotated_corners[3][1], 0],
                        [rotated_corners[0][0], rotated_corners[0][1], 0],
                    ]
                )
                colors.extend([color] * 8)
            else:
                rot = np.array(
                    [
                        [rot_data[0], rot_data[4], rot_data[8]],
                        [rot_data[1], rot_data[5], rot_data[9]],
                        [rot_data[2], rot_data[6], rot_data[10]],
                    ],
                    dtype="f4",
                )

                corners = np.array(
                    [
                        [0, 0, 0],
                        [vs, 0, 0],
                        [0, vs, 0],
                        [vs, vs, 0],
                        [0, 0, vs],
                        [vs, 0, vs],
                        [0, vs, vs],
                        [vs, vs, vs],
                    ],
                    dtype="f4",
                )

                cell_center = np.array([vs / 2, vs / 2, vs / 2], dtype="f4")
                world_offset = np.array(
                    [center + offset[0], center + offset[1], center + offset[2]],
                    dtype="f4",
                )
                rotated_corners = []
                for c in corners:
                    local = c - cell_center
                    rotated = rot @ local
                    world = rotated + cell_center + world_offset
                    rotated_corners.append(world)

                edges = [
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (1, 3),
                    (1, 5),
                    (2, 3),
                    (2, 6),
                    (3, 7),
                    (4, 5),
                    (4, 6),
                    (5, 7),
                    (6, 7),
                ]
                for i, j in edges:
                    positions.extend(
                        [
                            list(rotated_corners[i]),
                            list(rotated_corners[j]),
                        ]
                    )
                colors.extend([color] * 24)

        positions = np.array(positions, dtype="f4")
        colors = np.array(colors, dtype="f4")
        pos_buf = self.ctx.buffer(positions.tobytes())
        color_buf = self.ctx.buffer(colors.tobytes())
        self.grid_debug_vao = self.ctx.vertex_array(
            self.grid_debug_prog,
            [(pos_buf, "3f", "in_pos"), (color_buf, "3f", "in_color")],
        )
        self.grid_debug_bufs = [pos_buf, color_buf]

    def clear_mass(self):
        """Clear all grid mass buffers."""
        cells_per_grid = self.grid_size**self.dim
        total_cells = self.n_grids * cells_per_grid
        zeros = np.zeros(total_cells, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft_2d(self, src_tex, dst_tex, forward=True):
        """Run batched 2D FFT with Z as batch dimension."""
        gs = self.grid_size
        ng = self.n_grids
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        curr_src = src_tex
        curr_dst = dst_tex

        for axis in [0, 1]:
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis
            self.prog_fft["gridSize"] = gs

            for stage in range(num_stages + 1):
                self.prog_fft["stage"] = stage

                curr_src.bind_to_image(0, read=True, write=False)
                curr_dst.bind_to_image(1, read=False, write=True)

                self.prog_fft.run(max(1, gs // 8), max(1, gs // 8), ng)
                self.ctx.memory_barrier()

                curr_src, curr_dst = curr_dst, curr_src

        return curr_src

    def run_fft_3d_batched(self, src_tex, dst_tex, forward=True):
        """Run batched 3D FFT on packed texture (X, Y, Z*n_grids).

        Each grid occupies Z slices [i*gs : (i+1)*gs]. FFT operates within
        each grid's slice independently, avoiding spectral leakage between grids.
        """
        gs = self.grid_size
        ng = self.n_grids
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        curr_src = src_tex
        curr_dst = dst_tex

        for axis in [0, 1, 2]:
            self.prog_fft_batched["direction"] = direction
            self.prog_fft_batched["axis"] = axis
            self.prog_fft_batched["gridSize"] = gs
            self.prog_fft_batched["nGrids"] = ng

            for stage in range(num_stages + 1):
                self.prog_fft_batched["stage"] = stage

                curr_src.bind_to_image(0, read=True, write=False)
                curr_dst.bind_to_image(1, read=False, write=True)

                self.prog_fft_batched.run(max(1, gs // 8), max(1, gs // 8), ng)
                self.ctx.memory_barrier()

                curr_src, curr_dst = curr_dst, curr_src

        return curr_src

    def step_2d(self):
        """Simulation step for 2D."""
        gs = self.grid_size
        ng = self.n_grids
        n_particles = self.num_particles
        dispatch = (max(1, gs // 8), max(1, gs // 8), ng)

        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.rotation_buf.bind_to_storage_buffer(3)
        self.prog_mass["gridSize"] = gs
        self.prog_mass["nGrids"] = ng
        self.prog_mass["voxelSize"] = self.voxel_size
        self.prog_mass["worldSize"] = self.world_size
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        self.prog_complex["gridSize"] = gs
        self.prog_complex["nGrids"] = ng
        self.prog_complex.run(*dispatch)
        self.ctx.memory_barrier()

        spectrum_tex = self.run_fft_2d(
            self.complex_tex_a, self.complex_tex_b, forward=True
        )

        spectrum_tex.bind_to_image(0, read=True, write=True)
        self.prog_fourier["gridSize"] = gs
        self.prog_fourier["nGrids"] = ng
        self.prog_fourier.run(*dispatch)
        self.ctx.memory_barrier()

        spectrum_tex.bind_to_image(0, read=True, write=False)
        self.grad_comp_x.bind_to_image(1, read=False, write=True)
        self.grad_comp_y.bind_to_image(2, read=False, write=True)
        self.prog_greens["gridSize"] = gs
        self.prog_greens["nGrids"] = ng
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size
        self.prog_greens.run(*dispatch)
        self.ctx.memory_barrier()

        real_x = self.run_fft_2d(self.grad_comp_x, self.complex_tex_b, forward=False)
        real_y = self.run_fft_2d(self.grad_comp_y, self.complex_tex_a, forward=False)

        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        self.gradient_tex.bind_to_image(3, read=False, write=True)
        self.prog_pack["gridSize"] = gs
        self.prog_pack["nGrids"] = ng
        self.prog_pack.run(*dispatch)
        self.ctx.memory_barrier()

        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        self.prog_update["deltaTime"] = self.dt * self.timescale
        self.prog_update["gridSize"] = gs
        self.prog_update["nGrids"] = ng
        self.prog_update["worldSize"] = self.world_size
        self.prog_update["voxelSize"] = self.voxel_size
        self.prog_update["damping"] = self.cfg.get("damping", 1.0)
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

    def step_3d(self):
        """Simulation step for 3D - batched processing of all grids.

        Uses packed texture layout (X, Y, Z*n_grids) to process all grids
        in parallel, avoiding the O(n_grids) loop overhead.
        """
        gs = self.grid_size
        ng = self.n_grids
        n_particles = self.num_particles
        dispatch = (max(1, gs // 8), max(1, gs // 8), gs * ng)

        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.rotation_buf.bind_to_storage_buffer(3)
        set_uniform(self.prog_mass, "gridSize", gs)
        set_uniform(self.prog_mass, "nGrids", ng)
        set_uniform(self.prog_mass, "voxelSize", self.voxel_size)
        set_uniform(self.prog_mass, "worldSize", self.world_size)
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        set_uniform(self.prog_complex, "gridSize", gs)
        set_uniform(self.prog_complex, "nGrids", ng)
        self.prog_complex.run(*dispatch)
        self.ctx.memory_barrier()

        spectrum_tex = self.run_fft_3d_batched(
            self.complex_tex_a, self.complex_tex_b, forward=True
        )

        spectrum_tex.bind_to_image(0, read=True, write=True)
        set_uniform(self.prog_fourier, "gridSize", gs)
        set_uniform(self.prog_fourier, "nGrids", ng)
        self.prog_fourier.run(*dispatch)
        self.ctx.memory_barrier()

        spectrum_tex.bind_to_image(0, read=True, write=False)
        self.grad_comp_x.bind_to_image(1, read=False, write=True)
        self.grad_comp_y.bind_to_image(2, read=False, write=True)
        self.grad_comp_z.bind_to_image(3, read=False, write=True)
        set_uniform(self.prog_greens, "gridSize", gs)
        set_uniform(self.prog_greens, "nGrids", ng)
        set_uniform(self.prog_greens, "G", self.G)
        set_uniform(self.prog_greens, "worldSize", self.world_size)
        self.prog_greens.run(*dispatch)
        self.ctx.memory_barrier()

        real_x = self.run_fft_3d_batched(
            self.grad_comp_x, self.complex_tex_b, forward=False
        )
        real_y = self.run_fft_3d_batched(
            self.grad_comp_y, self.complex_tex_a, forward=False
        )
        real_z = self.run_fft_3d_batched(
            self.grad_comp_z, self.complex_tex_b, forward=False
        )

        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        real_z.bind_to_image(2, read=True, write=False)
        self.gradient_tex.bind_to_image(3, read=False, write=True)
        set_uniform(self.prog_pack, "gridSize", gs)
        set_uniform(self.prog_pack, "nGrids", ng)
        self.prog_pack.run(*dispatch)
        self.ctx.memory_barrier()

        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.rotation_buf.bind_to_storage_buffer(3)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        set_uniform(self.prog_update, "deltaTime", self.dt * self.timescale)
        set_uniform(self.prog_update, "gridSize", gs)
        set_uniform(self.prog_update, "nGrids", ng)
        set_uniform(self.prog_update, "worldSize", self.world_size)
        set_uniform(self.prog_update, "voxelSize", self.voxel_size)
        set_uniform(self.prog_update, "damping", self.cfg.get("damping", 1.0))
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

    def step(self):
        if self.dim == 2:
            self.step_2d()
        else:
            self.step_3d()

    def on_render(self, time, frame_time):
        if not self.paused:
            self.step()

        self.ctx.clear(0.02, 0.02, 0.05)
        self.ctx.enable(moderngl.BLEND)

        rad_x = np.radians(self.cam_rot_x)
        rad_y = np.radians(np.clip(self.cam_rot_y, -89, 89))

        eye_x = self.cam_dist * np.sin(rad_x) * np.cos(rad_y)
        eye_y = self.cam_dist * np.sin(rad_y)
        eye_z = self.cam_dist * np.cos(rad_x) * np.cos(rad_y)

        eye = np.array([eye_x, eye_y, eye_z], dtype="f4")
        target = np.array([0, 0, 0], dtype="f4")
        up = np.array([0, 1, 0], dtype="f4")

        m_view = create_look_at(eye, target, up)
        m_proj = create_perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        self.render_prog["m_view"].write(m_view.tobytes())
        self.render_prog["m_proj"].write(m_proj.tobytes())

        self.vao.render(moderngl.POINTS)

        if self.show_grid_debug:
            self.grid_debug_prog["m_view"].write(m_view.tobytes())
            self.grid_debug_prog["m_proj"].write(m_proj.tobytes())
            self.grid_debug_vao.render(moderngl.LINES)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.Q:
                self.wnd.close()

            elif key == keys.R:
                self.init_particles()
                print("Reset particles (GPU Perlin)")

            elif key == keys.NUMBER_1:
                self.init_particles_gpu(noise_scale=2.0, density_contrast=1.5)
                print("Preset 1: subtle clustering")

            elif key == keys.NUMBER_2:
                self.init_particles_gpu(noise_scale=4.0, density_contrast=2.5)
                print("Preset 2: medium clustering")

            elif key == keys.NUMBER_3:
                self.init_particles_gpu(noise_scale=8.0, density_contrast=4.0)
                print("Preset 3: tight clusters")

            elif key == keys.T:
                self.init_particles2()
                print("Reset particles (two galaxies)")

            elif key == keys.SPACE:
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Running'}")

            elif key == keys.BACKSLASH:
                self.timescale *= -1.0
                print(f"Timescale Inverted: {self.timescale:.4f}")

            elif key == keys.LEFT_BRACKET:
                self.timescale /= 1.5
                print(f"Timescale: {self.timescale:.4f}")

            elif key == keys.RIGHT_BRACKET:
                self.timescale *= 1.5
                print(f"Timescale: {self.timescale:.4f}")

            elif key == keys.G:
                self.show_grid_debug = not self.show_grid_debug
                print(f"Grid debug: {'ON' if self.show_grid_debug else 'OFF'}")


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
