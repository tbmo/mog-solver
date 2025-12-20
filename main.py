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
        self.wnd.key_event_func = self.key_event
        self.wnd.mouse_drag_event_func = self.mouse_drag_event
        self.wnd.mouse_scroll_event_func = self.mouse_scroll_event
        self.cfg = self.load_config()

        self.dim = self.cfg.get("dimensions", 2)
        if self.dim not in [2, 3]:
            raise NotImplementedError("Only 2D and 3D supported.")

        self.timescale = 1.0
        self.paused = False

        # Linear-Sparse-Grid parameters
        self.n_grids = int(self.cfg.get("n_grids", 16))
        self.grid_size = int(self.cfg["grid_size"])
        self.world_size = float(self.cfg["world_size"])
        self.voxel_size = self.world_size / self.grid_size
        self.num_particles = int(float(self.cfg["num_particles"]))
        self.G = self.cfg["G"]
        self.dt = self.cfg["dt"]

        # Precompute diagonal offsets
        self.offsets = self.compute_offsets()

        self.init_buffers()
        self.init_textures()
        self.init_shaders()
        self.init_particles()
        self.init_render()

        # Camera State
        self.cam_rot_x = 0.0
        self.cam_rot_y = 0.0
        self.cam_dist = 3.0
        self.mouse_pressed = False
        self.show_grid_debug = False

        effective_res = int(self.n_grids ** (1.0 / self.dim) * self.grid_size)
        total_cells = self.n_grids * (self.grid_size ** self.dim)
        equiv_cells = effective_res ** self.dim

        print(f"Linear-Sparse-Grid Config ({self.dim}D):")
        print(f"  n_grids: {self.n_grids}")
        print(f"  grid_size: {self.grid_size}^{self.dim}")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Equivalent single grid: {effective_res}^{self.dim} = {equiv_cells:,} cells")
        print(f"  Memory ratio: {total_cells / equiv_cells:.2f}x")

    def compute_offsets(self):
        """
        Compute diagonal offsets for each grid.
        For 2D: offset[i] = (i/n, i/n) * voxel_size
        For 3D: offset[i] = (i/n, i/n, i/n) * voxel_size
        """
        offsets = np.zeros((self.n_grids, 4), dtype="f4")  # vec4 for alignment
        for i in range(self.n_grids):
            frac = i / self.n_grids
            if self.dim == 2:
                offsets[i] = [frac * self.voxel_size, frac * self.voxel_size, 0, 0]
            else:  # 3D
                offsets[i] = [frac * self.voxel_size, frac * self.voxel_size, frac * self.voxel_size, 0]
        return offsets

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
        # Particle buffers
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

        # Offset buffer for GPU access (vec4 aligned)
        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())

        # Mass buffer: one contiguous buffer for all grids
        # Layout: [grid0_cells..., grid1_cells..., ...]
        cells_per_grid = self.grid_size ** self.dim
        total_cells = self.n_grids * cells_per_grid
        self.mass_buf = self.ctx.buffer(reserve=total_cells * 4)

    def init_textures(self):
        """
        Create textures for each grid independently to avoid spectral leakage.
        For 2D: Use 3D texture with Z as batch (FFT only on X,Y)
        For 3D: Use array of 3D textures, one per grid
        """
        gs = self.grid_size
        ng = self.n_grids

        if self.dim == 2:
            # 2D: Z dimension is batch, no spectral leakage since FFT only on X,Y
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
            # 3D: Create separate textures for EACH grid to avoid spectral leakage
            size = (gs, gs, gs)

            def create_tex_array(channels):
                return [self.ctx.texture3d(size, channels, dtype="f4") for _ in range(ng)]

            self.complex_tex_a = create_tex_array(2)
            self.complex_tex_b = create_tex_array(2)
            self.complex_tex_c = create_tex_array(2)

            self.grad_comp_x = create_tex_array(2)
            self.grad_comp_y = create_tex_array(2)
            self.grad_comp_z = create_tex_array(2)

            self.gradient_tex = create_tex_array(4)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.wnd.mouse_states.left:
            self.cam_rot_x -= dx * 0.5
            self.cam_rot_y += dy * 0.5

    def mouse_scroll_event(self, x_offset, y_offset):
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
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))
        self.prog_fourier = self.ctx.compute_shader(load("fourier.glsl"))
        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))

    def init_particles(self):
        """Initialize particles with multi-octave noise."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        num_octaves = np.random.randint(2, 6)
        base_freq = np.random.uniform(0.5, 4.0)
        lacunarity = np.random.uniform(1.5, 3.0)
        persistence = np.random.uniform(0.3, 0.7)
        noise_strength = np.random.uniform(0.2, 0.5) * self.world_size

        offset = np.random.uniform(0, 1000, size=self.dim)

        def perlin_octaves(coords):
            total = np.zeros(len(coords))
            freq = base_freq
            amp = 1.0
            max_amp = 0.0

            for _ in range(num_octaves):
                noise = np.ones(len(coords))
                for d in range(coords.shape[1]):
                    noise *= np.sin(freq * coords[:, d] + offset[d])
                    noise += np.cos(freq * 1.7 * coords[:, d] + offset[d] * 0.7)
                total += noise * amp
                max_amp += amp
                freq *= lacunarity
                amp *= persistence

            return total / max_amp

        if np.random.random() < 0.5:
            base_pos = (
                np.random.uniform(0.1, 0.9, (self.num_particles, self.dim))
                * self.world_size
            )
        else:
            n_side = int(np.ceil(self.num_particles ** (1.0 / self.dim)))
            grid = np.meshgrid(
                *[np.linspace(0.1, 0.9, n_side) for _ in range(self.dim)]
            )
            base_pos = (
                np.stack([g.flatten() for g in grid], axis=1)[: self.num_particles]
                * self.world_size
            )
            base_pos += np.random.normal(0, self.world_size * 0.01, base_pos.shape)

        for d in range(self.dim):
            noise_coords = base_pos / self.world_size * base_freq + offset[d] * 100
            displacement = perlin_octaves(noise_coords) * noise_strength
            base_pos[:, d] += displacement

        base_pos = np.clip(base_pos, 0.05 * self.world_size, 0.95 * self.world_size)

        pos[:, : self.dim] = base_pos
        pos[:, 3] = self.cfg["particle_mass"]

        self.vel_buf.write(vel.tobytes())
        self.pos_buf.write(pos.tobytes())

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

        # Debug grid line rendering
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
        """Create grid line geometry for debug visualization."""
        # Generate distinct colors for each grid using golden ratio hue
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

        positions = []
        colors = []
        gs = self.grid_size
        vs = self.voxel_size
        ws = self.world_size

        for grid_idx in range(self.n_grids):
            # Golden ratio color generation for distinct colors
            hue = (grid_idx * 0.618033988749895) % 1.0
            r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
            color = (r, g, b)
            offset = self.offsets[grid_idx][:self.dim]

            if self.dim == 2:
                # Horizontal lines
                for i in range(gs + 1):
                    y = (i * vs + offset[1]) % ws
                    positions.extend([[0, y, 0], [ws, y, 0]])
                    colors.extend([color, color])
                # Vertical lines
                for i in range(gs + 1):
                    x = (i * vs + offset[0]) % ws
                    positions.extend([[x, 0, 0], [x, ws, 0]])
                    colors.extend([color, color])
            else:  # 3D
                # X-direction lines (along X axis)
                for iy in range(gs + 1):
                    for iz in range(gs + 1):
                        y = (iy * vs + offset[1]) % ws
                        z = (iz * vs + offset[2]) % ws
                        positions.extend([[0, y, z], [ws, y, z]])
                        colors.extend([color, color])
                # Y-direction lines
                for ix in range(gs + 1):
                    for iz in range(gs + 1):
                        x = (ix * vs + offset[0]) % ws
                        z = (iz * vs + offset[2]) % ws
                        positions.extend([[x, 0, z], [x, ws, z]])
                        colors.extend([color, color])
                # Z-direction lines
                for ix in range(gs + 1):
                    for iy in range(gs + 1):
                        x = (ix * vs + offset[0]) % ws
                        y = (iy * vs + offset[1]) % ws
                        positions.extend([[x, y, 0], [x, y, ws]])
                        colors.extend([color, color])

        positions = np.array(positions, dtype="f4")
        colors = np.array(colors, dtype="f4")

        self.grid_debug_pos_buf = self.ctx.buffer(positions.tobytes())
        self.grid_debug_color_buf = self.ctx.buffer(colors.tobytes())
        self.grid_debug_vao = self.ctx.vertex_array(
            self.grid_debug_prog,
            [
                (self.grid_debug_pos_buf, "3f", "in_pos"),
                (self.grid_debug_color_buf, "3f", "in_color"),
            ],
        )
        self.grid_debug_vertex_count = len(positions)

    def clear_mass(self):
        """Clear all grid mass buffers."""
        cells_per_grid = self.grid_size ** self.dim
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

        for axis in [0, 1]:  # X, Y only
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

    def run_fft_3d_single(self, src_tex, dst_tex, scratch_tex, forward=True):
        """Run 3D FFT on a single grid texture."""
        gs = self.grid_size
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        curr_src = src_tex
        curr_dst = dst_tex

        for axis in [0, 1, 2]:  # X, Y, Z
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis
            self.prog_fft["gridSize"] = gs

            for stage in range(num_stages + 1):
                self.prog_fft["stage"] = stage

                curr_src.bind_to_image(0, read=True, write=False)
                curr_dst.bind_to_image(1, read=False, write=True)

                self.prog_fft.run(max(1, gs // 8), max(1, gs // 8), gs)
                self.ctx.memory_barrier()

                curr_src, curr_dst = curr_dst, curr_src

        return curr_src

    def step_2d(self):
        """Simulation step for 2D."""
        gs = self.grid_size
        ng = self.n_grids
        n_particles = self.num_particles
        dispatch = (max(1, gs // 8), max(1, gs // 8), ng)

        # Step A: Scatter
        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.prog_mass["gridSize"] = gs
        self.prog_mass["nGrids"] = ng
        self.prog_mass["voxelSize"] = self.voxel_size
        self.prog_mass["worldSize"] = self.world_size
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # Step B.1: Mass -> Complex
        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        self.prog_complex["gridSize"] = gs
        self.prog_complex["nGrids"] = ng
        self.prog_complex.run(*dispatch)
        self.ctx.memory_barrier()

        # Step B.2: Forward FFT
        spectrum_tex = self.run_fft_2d(self.complex_tex_a, self.complex_tex_b, forward=True)

        # Step B.3: Zero DC
        spectrum_tex.bind_to_image(0, read=True, write=True)
        self.prog_fourier["gridSize"] = gs
        self.prog_fourier["nGrids"] = ng
        self.prog_fourier.run(*dispatch)
        self.ctx.memory_barrier()

        # Step B.4: Green's function
        spectrum_tex.bind_to_image(0, read=True, write=False)
        self.grad_comp_x.bind_to_image(1, read=False, write=True)
        self.grad_comp_y.bind_to_image(2, read=False, write=True)
        self.prog_greens["gridSize"] = gs
        self.prog_greens["nGrids"] = ng
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size
        self.prog_greens.run(*dispatch)
        self.ctx.memory_barrier()

        # Step B.5: Inverse FFT
        real_x = self.run_fft_2d(self.grad_comp_x, self.complex_tex_b, forward=False)
        real_y = self.run_fft_2d(self.grad_comp_y, self.complex_tex_a, forward=False)

        # Step B.6: Pack
        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        self.gradient_tex.bind_to_image(3, read=False, write=True)
        self.prog_pack["gridSize"] = gs
        self.prog_pack["nGrids"] = ng
        self.prog_pack.run(*dispatch)
        self.ctx.memory_barrier()

        # Step C & D: Gather + Integrate
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        self.prog_update["deltaTime"] = self.dt * self.timescale
        self.prog_update["gridSize"] = gs
        self.prog_update["nGrids"] = ng
        self.prog_update["worldSize"] = self.world_size
        self.prog_update["voxelSize"] = self.voxel_size
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

    def step_3d(self):
        """Simulation step for 3D - processes grids independently."""
        gs = self.grid_size
        ng = self.n_grids
        n_particles = self.num_particles
        dispatch_single = (max(1, gs // 8), max(1, gs // 8), gs)

        # Step A: Scatter (all grids at once)
        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        set_uniform(self.prog_mass, "gridSize", gs)
        set_uniform(self.prog_mass, "nGrids", ng)
        set_uniform(self.prog_mass, "voxelSize", self.voxel_size)
        set_uniform(self.prog_mass, "worldSize", self.world_size)
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # Process each grid independently for FFT to avoid spectral leakage
        cells_per_grid = gs * gs * gs

        for grid_idx in range(ng):
            # Step B.1: Mass -> Complex for this grid
            self.mass_buf.bind_to_storage_buffer(0)
            self.complex_tex_a[grid_idx].bind_to_image(1, read=False, write=True)
            set_uniform(self.prog_complex, "gridSize", gs)
            set_uniform(self.prog_complex, "nGrids", ng)
            set_uniform(self.prog_complex, "currentGrid", grid_idx)
            self.prog_complex.run(*dispatch_single)
            self.ctx.memory_barrier()

            # Step B.2: Forward FFT
            spectrum_tex = self.run_fft_3d_single(
                self.complex_tex_a[grid_idx],
                self.complex_tex_b[grid_idx],
                self.complex_tex_c[grid_idx],
                forward=True
            )

            # Step B.3: Zero DC
            spectrum_tex.bind_to_image(0, read=True, write=True)
            set_uniform(self.prog_fourier, "gridSize", gs)
            set_uniform(self.prog_fourier, "nGrids", ng)
            self.prog_fourier.run(*dispatch_single)
            self.ctx.memory_barrier()

            # Step B.4: Green's function
            spectrum_tex.bind_to_image(0, read=True, write=False)
            self.grad_comp_x[grid_idx].bind_to_image(1, read=False, write=True)
            self.grad_comp_y[grid_idx].bind_to_image(2, read=False, write=True)
            self.grad_comp_z[grid_idx].bind_to_image(3, read=False, write=True)
            set_uniform(self.prog_greens, "gridSize", gs)
            set_uniform(self.prog_greens, "G", self.G)
            set_uniform(self.prog_greens, "worldSize", self.world_size)
            self.prog_greens.run(*dispatch_single)
            self.ctx.memory_barrier()

            # Step B.5: Inverse FFT
            real_x = self.run_fft_3d_single(
                self.grad_comp_x[grid_idx],
                self.complex_tex_b[grid_idx],
                self.complex_tex_c[grid_idx],
                forward=False
            )
            real_y = self.run_fft_3d_single(
                self.grad_comp_y[grid_idx],
                self.complex_tex_a[grid_idx],
                self.complex_tex_c[grid_idx],
                forward=False
            )
            real_z = self.run_fft_3d_single(
                self.grad_comp_z[grid_idx],
                self.complex_tex_c[grid_idx],
                self.complex_tex_b[grid_idx],
                forward=False
            )

            # Step B.6: Pack
            real_x.bind_to_image(0, read=True, write=False)
            real_y.bind_to_image(1, read=True, write=False)
            real_z.bind_to_image(2, read=True, write=False)
            self.gradient_tex[grid_idx].bind_to_image(3, read=False, write=True)
            set_uniform(self.prog_pack, "gridSize", gs)
            set_uniform(self.prog_pack, "nGrids", ng)
            self.prog_pack.run(*dispatch_single)
            self.ctx.memory_barrier()

        # Step C & D: Gather from all grids + Integrate
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        # Bind all gradient textures for gather
        for grid_idx in range(ng):
            self.gradient_tex[grid_idx].bind_to_image(grid_idx, read=True, write=False)
        set_uniform(self.prog_update, "deltaTime", self.dt * self.timescale)
        set_uniform(self.prog_update, "gridSize", gs)
        set_uniform(self.prog_update, "nGrids", ng)
        set_uniform(self.prog_update, "worldSize", self.world_size)
        set_uniform(self.prog_update, "voxelSize", self.voxel_size)
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

        # Debug grid rendering
        if self.show_grid_debug:
            self.grid_debug_prog["m_view"].write(m_view.tobytes())
            self.grid_debug_prog["m_proj"].write(m_proj.tobytes())
            self.grid_debug_vao.render(moderngl.LINES)

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.Q:
                self.wnd.close()

            elif key == keys.R:
                self.init_particles()
                print("Reset particles")

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

            elif key == keys.NUMBER_1:
                self.show_grid_debug = not self.show_grid_debug
                print(f"Grid debug: {'ON' if self.show_grid_debug else 'OFF'}")


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
