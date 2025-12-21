"""
Linear-Sparse-Grid N-Body Solver (3D)

Achieves high effective resolution by stacking multiple low-resolution grids
with diagonal spatial offsets, scaling linearly O(n) rather than quadratically O(n²).
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
        params = f"gs{self.grid_size}_ws{self.world_size}_ng{self.n_grids}_s{self.seed}"
        return hashlib.md5(params.encode()).hexdigest()[:12]

    def _load_cache(self):
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
        V = self.voxel_size
        W = self.world_size
        G = self.grid_size

        char_radius = W / (2 * np.sqrt(3))
        max_rot_displacement = char_radius * np.pi
        max_trans_displacement = V * np.sqrt(3) / 2

        grid_factor = min(1.0, 60 / self.n_grids)
        coarseness = 32 / G

        base_rot = 0.6
        coarse_adjust = 0.15 * (coarseness - 1)
        count_adjust = 0.1 * (1 - grid_factor)

        w_rot = np.clip(base_rot - coarse_adjust + count_adjust, 0.3, 0.8)
        w_trans = 1 - w_rot

        print("SE(3) Sampler Auto-Tuning:")
        print(f"  voxel_size: {V:.2f}")
        print(f"  char_radius: {char_radius:.2f}")
        print(f"  max_rot_displacement: {max_rot_displacement:.2f}")
        print(f"  max_trans_displacement: {max_trans_displacement:.2f}")
        print(f"  coarseness factor: {coarseness:.2f} (32/G)")
        print(f"  grid count factor: {grid_factor:.2f}")
        print(f"  → weights: rotation={w_rot:.2f}, translation={w_trans:.2f}")

        return w_rot, w_trans

    def quat_to_matrix(self, q):
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
        q1, off1 = c1
        q2, off2 = c2

        dot = np.abs(np.dot(q1, q2))
        rot_dist = 2 * np.arccos(np.clip(dot, 0, 1)) / np.pi

        diff = off1 - off2
        diff = diff - self.voxel_size * np.round(diff / self.voxel_size)
        trans_dist = np.linalg.norm(diff) / (self.voxel_size * np.sqrt(3) / 2)

        return self.w_rot * rot_dist + self.w_trans * trans_dist

    def min_distance_to_set(self, config):
        if len(self._configs) == 0:
            return 1.0
        return min(self.config_distance(config, c) for c in self._configs)

    def get_icosahedral_quaternions(self):
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
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 3, 2, 0), (1, 0, 3, 2),
            (2, 0, 1, 3), (2, 3, 0, 1), (2, 1, 3, 0),
            (3, 1, 0, 2), (3, 0, 2, 1), (3, 2, 1, 0),
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
                        q = [0.5 * vals[0], 0.5 * s1 * vals[1], 0.5 * s2 * vals[2], 0.5 * s3 * vals[3]]
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
        q = q / np.linalg.norm(q)
        for i in range(4):
            if abs(q[i]) > 1e-6:
                return np.array(-q if q[i] < 0 else q, dtype="f4")
        return np.array(q, dtype="f4")

    def _random_config(self):
        q = self.rng.standard_normal(4).astype("f4")
        q = self._canonicalize_quat(q)
        off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
        return (q, off)

    def _optimize_offset_for_rotation(self, q, n_candidates=200):
        best_off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
        best_dist = self.min_distance_to_set((q, best_off))

        for _ in range(n_candidates):
            off = self.rng.uniform(0, self.voxel_size, 3).astype("f4")
            d = self.min_distance_to_set((q, off))
            if d > best_dist:
                best_dist = d
                best_off = off

        return best_off, best_dist

    def get_transforms(self, n=None, verbose=False, optimize=True):
        if n is None:
            n = self.n_grids

        already_cached = self._max_computed >= n

        if optimize and not already_cached:
            if verbose:
                print("  Running energy optimization...")
            self.optimize_energy(n_iterations=500, verbose=verbose)

        rotations = np.zeros((n, 3, 3), dtype="f4")
        offsets = np.zeros((n, 3), dtype="f4")

        for i, (q, off) in enumerate(self._configs[:n]):
            rotations[i] = self.quat_to_matrix(q)
            offsets[i] = off

        return rotations, offsets

    def optimize_energy(self, n_iterations=500, verbose=True):
        if len(self._configs) == 0:
            return

        n = len(self._configs)
        quats = np.array([c[0] for c in self._configs])
        offs = np.array([c[1] for c in self._configs])

        lr_quat = 0.002
        lr_off = 0.005 * self.voxel_size

        best_energy = float("inf")
        best_quats = quats.copy()
        best_offs = offs.copy()

        for iteration in range(n_iterations):
            quat_grads = np.zeros_like(quats)
            off_grads = np.zeros_like(offs)
            total_energy = 0

            for i in range(n):
                for j in range(i + 1, n):
                    dot = np.dot(quats[i], quats[j])
                    sign = np.sign(dot) if dot != 0 else 1
                    dot_abs = np.abs(dot)
                    rot_dist = 2 * np.arccos(np.clip(dot_abs, 0, 0.9999))

                    off_diff = offs[i] - offs[j]
                    off_diff = off_diff - self.voxel_size * np.round(off_diff / self.voxel_size)
                    trans_dist = np.linalg.norm(off_diff) + 1e-8

                    dist = self.w_rot * rot_dist / np.pi + self.w_trans * trans_dist / (self.voxel_size * 0.866)
                    dist = max(dist, 0.01)

                    total_energy += 1.0 / dist
                    force = 1.0 / (dist * dist)

                    quat_diff = quats[i] - sign * quats[j]
                    quat_diff_norm = np.linalg.norm(quat_diff) + 1e-8
                    quat_grads[i] += force * quat_diff / quat_diff_norm
                    quat_grads[j] -= force * sign * quat_diff / quat_diff_norm

                    off_grads[i] += force * off_diff / trans_dist
                    off_grads[j] -= force * off_diff / trans_dist

            quats += lr_quat * quat_grads
            offs += lr_off * off_grads

            quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

            for i in range(n):
                for k in range(4):
                    if abs(quats[i, k]) > 1e-6:
                        if quats[i, k] < 0:
                            quats[i] = -quats[i]
                        break

            offs = offs % self.voxel_size

            if total_energy < best_energy:
                best_energy = total_energy
                best_quats = quats.copy()
                best_offs = offs.copy()

            lr_quat *= 0.998
            lr_off *= 0.998

            if verbose and (iteration + 1) % 100 == 0:
                print(f"  Energy iter {iteration + 1}: energy={total_energy:.2f}, best={best_energy:.2f}")

        self._configs = [(best_quats[i], best_offs[i]) for i in range(n)]
        self._save_cache()

        if verbose:
            print(f"  Energy optimization complete. Best energy: {best_energy:.2f}")

    def analyze_coverage(self, n=None):
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

        solver = self.cfg.get("solver", "convolution")
        self.use_convolution = solver == "convolution"

        self.init_buffers()
        self.init_textures()
        self.init_shaders()
        self.init_greens_kernel()
        self.init_particles()
        self.init_render()

        self.cam_rot_x = 0.0
        self.cam_rot_y = 0.0
        self.cam_dist = 3.0
        self.mouse_pressed = False
        self.show_grid_debug = False
        self.show_particles = True

        effective_res = self.grid_size * self.n_grids
        total_cells = self.n_grids * (self.grid_size ** 3)
        equiv_cells = effective_res ** 3

        solver_mode = "Direct Convolution" if self.use_convolution else "FFT"
        print(f"Linear-Sparse-Grid Config (3D):")
        print(f"  Solver: {solver_mode}")
        print(f"  world_size: {self.world_size}")
        print(f"  num_particles: {self.num_particles:,}")
        print(f"  n_grids: {self.n_grids}")
        print(f"  grid_size: {self.grid_size}^3")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Equivalent single grid: {effective_res}^3 = {equiv_cells:,} cells")
        print(f"  Memory savings: {equiv_cells / total_cells:.1f}x")

    def compute_transforms(self):
        offsets = np.zeros((self.n_grids, 4), dtype="f4")
        rotations = np.zeros((self.n_grids, 12), dtype="f4")

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
            "grid_size": 16,
            "world_size": 100.0,
            "num_particles": 100000,
            "G": 1.0,
            "dt": 0.00001,
            "particle_mass": 1.0,
            "solver": "convolution",
        }

    def init_buffers(self):
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())
        self.rotation_buf = self.ctx.buffer(self.rotations.tobytes())

        cells_per_grid = self.grid_size ** 3
        total_cells = self.n_grids * cells_per_grid
        self.mass_buf = self.ctx.buffer(reserve=total_cells * 4)

    def init_textures(self):
        gs = self.grid_size
        ng = self.n_grids
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
            header = f"#version 460\n#define DIM 3\n#define N_GRIDS {self.n_grids}\n#define GRID_SIZE {self.grid_size}\n"
            src = src.replace("#version 460", "")
            return header + src

        self.prog_mass = self.ctx.compute_shader(load("mass.glsl"))
        self.prog_complex = self.ctx.compute_shader(load("complex.glsl"))
        self.prog_fft_batched = self.ctx.compute_shader(load("fft_batched.glsl"))
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))
        self.prog_fourier = self.ctx.compute_shader(load("fourier.glsl"))
        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))
        self.prog_init_particles = self.ctx.compute_shader(load("init_particles.glsl"))
        self.prog_convolve = self.ctx.compute_shader(load("convolve.glsl"))

    def init_greens_kernel(self):
        """Precompute the real-space Green's gradient kernel for 3D."""
        gs = self.grid_size
        voxel = self.voxel_size

        kernel = np.zeros((gs, gs, gs, 3), dtype="f4")

        for iz in range(gs):
            for iy in range(gs):
                for ix in range(gs):
                    dx = ix if ix <= gs // 2 else ix - gs
                    dy = iy if iy <= gs // 2 else iy - gs
                    dz = iz if iz <= gs // 2 else iz - gs

                    rx = dx * voxel
                    ry = dy * voxel
                    rz = dz * voxel
                    r2 = rx * rx + ry * ry + rz * rz
                    r = np.sqrt(r2)

                    if r > 1e-10:
                        # For gravity: attractive force points toward source
                        factor = 1.0 / (4.0 * np.pi * r2 * r)
                        kernel[iz, iy, ix, 0] = rx * factor
                        kernel[iz, iy, ix, 1] = ry * factor
                        kernel[iz, iy, ix, 2] = rz * factor

        print(f"Green's kernel (3D): {gs}x{gs}x{gs}x3, voxel={voxel:.2f}")

        self.green_kernel_buf = self.ctx.buffer(kernel.tobytes())
        print(f"  Kernel buffer size: {kernel.nbytes / 1024:.1f} KB")

    def init_particles(self):
        self.init_particles_gpu(noise_scale=4.0, density_contrast=2.0, seed=None)

    def init_particles_gpu(self, noise_scale=4.0, density_contrast=2.0, seed=None, spawn_buffer=None):
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

        print(f"GPU init: {n_particles:,} particles, scale={noise_scale}, contrast={density_contrast}, buffer={spawn_buffer:.0%}")

    def init_particles2(self):
        """Initialize two colliding galaxies."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        center = self.world_size / 2.0
        n_per_galaxy = self.num_particles // 2

        for i, (offset, v_bulk) in enumerate([
            (np.array([-0.2, 0.0, 0.0]), np.array([0.3, 0.1, 0.0])),
            (np.array([0.2, 0.0, 0.0]), np.array([-0.3, -0.1, 0.0])),
        ]):
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

            spin_axis = np.array([0.3, 0.7, 0.5]) if i == 0 else np.array([-0.5, 0.2, 0.8])
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
        self.vao = self.ctx.vertex_array(self.render_prog, [(self.pos_buf, "4f", "in_pos")])

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

    def init_grid_debug_geometry(self, cells_per_axis=3, max_grids=None):
        def hsv_to_rgb(h, s, v):
            i = int(h * 6)
            f = h * 6 - i
            p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
            i = i % 6
            if i == 0: return (v, t, p)
            if i == 1: return (q, v, p)
            if i == 2: return (p, v, t)
            if i == 3: return (p, q, v)
            if i == 4: return (t, p, v)
            return (v, p, q)

        vs = self.voxel_size
        center = self.world_size / 2.0
        half_extent = cells_per_axis // 2

        positions = []
        colors = []

        n_to_draw = max_grids if max_grids else self.n_grids

        for grid_idx in range(min(n_to_draw, self.n_grids)):
            hue = (grid_idx * 0.618033988749895) % 1.0
            r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
            color = (r, g, b)

            offset = self.offsets[grid_idx][:3]
            rot_data = self.rotations[grid_idx]

            rot = np.array([
                [rot_data[0], rot_data[4], rot_data[8]],
                [rot_data[1], rot_data[5], rot_data[9]],
                [rot_data[2], rot_data[6], rot_data[10]],
            ], dtype="f4")

            world_offset = np.array([center + offset[0], center + offset[1], center + offset[2]], dtype="f4")

            for cx in range(-half_extent, half_extent + 1):
                for cy in range(-half_extent, half_extent + 1):
                    for cz in range(-half_extent, half_extent + 1):
                        cell_origin = np.array([cx * vs, cy * vs, cz * vs], dtype="f4")

                        corners = np.array([
                            cell_origin + [0, 0, 0],
                            cell_origin + [vs, 0, 0],
                            cell_origin + [0, vs, 0],
                            cell_origin + [vs, vs, 0],
                            cell_origin + [0, 0, vs],
                            cell_origin + [vs, 0, vs],
                            cell_origin + [0, vs, vs],
                            cell_origin + [vs, vs, vs],
                        ], dtype="f4")

                        rotated_corners = []
                        for c in corners:
                            rotated = rot @ c
                            world = rotated + world_offset
                            rotated_corners.append(world)

                        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
                        for i, j in edges:
                            positions.extend([list(rotated_corners[i]), list(rotated_corners[j])])
                        colors.extend([color] * 24)

        positions = np.array(positions, dtype="f4")
        colors = np.array(colors, dtype="f4")

        print(f"Grid debug: {len(positions)} vertices for {n_to_draw} grids × {cells_per_axis}^3 cells")

        pos_buf = self.ctx.buffer(positions.tobytes())
        color_buf = self.ctx.buffer(colors.tobytes())
        self.grid_debug_vao = self.ctx.vertex_array(
            self.grid_debug_prog,
            [(pos_buf, "3f", "in_pos"), (color_buf, "3f", "in_color")],
        )
        self.grid_debug_bufs = [pos_buf, color_buf]

    def clear_mass(self):
        cells_per_grid = self.grid_size ** 3
        total_cells = self.n_grids * cells_per_grid
        zeros = np.zeros(total_cells, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft_3d_batched(self, src_tex, dst_tex, forward=True):
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

    def step_fft(self):
        """Simulation step using FFT-based solver."""
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

        spectrum_tex = self.run_fft_3d_batched(self.complex_tex_a, self.complex_tex_b, forward=True)

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

        real_x = self.run_fft_3d_batched(self.grad_comp_x, self.complex_tex_b, forward=False)
        real_y = self.run_fft_3d_batched(self.grad_comp_y, self.complex_tex_a, forward=False)
        real_z = self.run_fft_3d_batched(self.grad_comp_z, self.complex_tex_b, forward=False)

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

    def step_convolve(self):
        """Simulation step using direct convolution (no FFT)."""
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
        self.green_kernel_buf.bind_to_storage_buffer(1)
        self.gradient_tex.bind_to_image(0, read=False, write=True)
        set_uniform(self.prog_convolve, "gridSize", gs)
        set_uniform(self.prog_convolve, "nGrids", ng)
        set_uniform(self.prog_convolve, "G", self.G)
        set_uniform(self.prog_convolve, "worldSize", self.world_size)
        self.prog_convolve.run(*dispatch)
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
        if self.use_convolution:
            self.step_convolve()
        else:
            self.step_fft()

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

        if self.show_particles:
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

            elif key == keys.H:
                self.show_particles = not self.show_particles
                print(f"Particle rendering: {'ON' if self.show_particles else 'OFF'}")

            elif key == keys.C:
                self.use_convolution = not self.use_convolution
                mode = "Direct Convolution" if self.use_convolution else "FFT"
                print(f"Solver: {mode}")


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
