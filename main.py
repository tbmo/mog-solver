"""
Linear-Sparse-Grid N-Body Solver

Achieves high effective resolution by stacking multiple low-resolution grids
with diagonal spatial offsets, scaling linearly O(n) rather than quadratically O(n²).
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import yaml


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
        if self.dim != 2:
            raise NotImplementedError("Linear-Sparse-Grid currently supports 2D only.")

        self.timescale = 1.0
        self.paused = False

        # Linear-Sparse-Grid parameters
        self.n_grids = int(self.cfg.get("n_grids", 16))
        self.grid_size = int(self.cfg["grid_size"])  # Base resolution (small)
        self.world_size = float(self.cfg["world_size"])
        self.voxel_size = self.world_size / self.grid_size
        self.num_particles = int(float(self.cfg["num_particles"]))
        self.G = self.cfg["G"]
        self.dt = self.cfg["dt"]

        # Precompute diagonal offsets: (i/n_grids, i/n_grids) normalized to cell width
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

        print(f"Linear-Sparse-Grid Config:")
        print(f"  n_grids: {self.n_grids}")
        print(f"  grid_size: {self.grid_size}x{self.grid_size}")
        print(f"  Total cells: {self.n_grids * self.grid_size * self.grid_size}")
        print(f"  Equivalent single grid: {int(np.sqrt(self.n_grids) * self.grid_size)}x{int(np.sqrt(self.n_grids) * self.grid_size)}")

    def compute_offsets(self):
        """
        Compute diagonal offsets for each grid.
        Offsets are in world-space units (fraction of voxel_size).
        offset[i] = (i/n_grids, i/n_grids) * voxel_size

        The offsets uniformly sample within one cell, starting from 0.
        Since the grid is periodic, this creates uniform sub-cell sampling.
        """
        offsets = np.zeros((self.n_grids, 2), dtype="f4")
        for i in range(self.n_grids):
            # Offset from 0 to (n_grids-1)/n_grids of a voxel
            frac = i / self.n_grids
            offsets[i] = [frac * self.voxel_size, frac * self.voxel_size]
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

        # Offset buffer for GPU access
        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())

        # Mass buffer: [n_grids, grid_size, grid_size] flattened
        total_cells = self.n_grids * self.grid_size * self.grid_size
        self.mass_buf = self.ctx.buffer(reserve=total_cells * 4)

    def init_textures(self):
        """
        Create 3D textures where Z-dimension is the grid index (batch dimension).
        Shape: [grid_size, grid_size, n_grids]
        """
        gs = self.grid_size
        ng = self.n_grids

        # 3D textures with n_grids as depth
        size = (gs, gs, ng)

        def create_tex(channels):
            return self.ctx.texture3d(size, channels, dtype="f4")

        # Mass density per grid
        self.mass_tex = create_tex(1)

        # FFT ping-pong textures (complex = 2 channels)
        self.complex_tex_a = create_tex(2)
        self.complex_tex_b = create_tex(2)

        # Force component textures (complex in Fourier space)
        self.grad_comp_x = create_tex(2)
        self.grad_comp_y = create_tex(2)

        # Final force field per grid (real space, 4 channels: fx, fy, 0, mag)
        self.gradient_tex = create_tex(4)

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

    def clear_mass(self):
        """Clear all grid mass buffers."""
        total_cells = self.n_grids * self.grid_size * self.grid_size
        zeros = np.zeros(total_cells, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft(self, src_tex, dst_tex, forward=True):
        """
        Run batched 2D FFT on 3D texture where Z is batch dimension.
        FFT operates on X and Y axes only, preserving Z (grid index).
        """
        gs = self.grid_size
        ng = self.n_grids
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        curr_src = src_tex
        curr_dst = dst_tex

        # FFT along X axis, then Y axis (Z is batch, unchanged)
        for axis in range(2):  # Only X and Y for 2D FFT
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis
            self.prog_fft["tensorDimensions"] = (gs, gs, ng)

            for stage in range(num_stages + 1):
                self.prog_fft["stage"] = stage

                curr_src.bind_to_image(0, read=True, write=False)
                curr_dst.bind_to_image(1, read=False, write=True)

                # Dispatch: (gs/8, gs/8, n_grids) - process all grids in parallel
                self.prog_fft.run(gs // 8, gs // 8, ng)
                self.ctx.memory_barrier()

                curr_src, curr_dst = curr_dst, curr_src

        return curr_src

    def step(self):
        gs = self.grid_size
        ng = self.n_grids
        dims = (gs, gs, ng)
        n_particles = self.num_particles

        def dispatch_grid():
            return (gs // 8, gs // 8, ng)

        # Step A: Scatter (Mass Assignment) - particles to all grids with offsets
        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.prog_mass["tensorDimensions"] = dims
        self.prog_mass["voxelSize"] = self.voxel_size
        self.prog_mass["worldSize"] = self.world_size
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # Step B.1: Mass buffer -> Complex texture
        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        self.prog_complex["tensorDimensions"] = dims
        self.prog_complex.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # Step B.2: Forward FFT (batched across all grids)
        spectrum_tex = self.run_fft(self.complex_tex_a, self.complex_tex_b, forward=True)

        # Step B.3: Zero DC mode
        spectrum_tex.bind_to_image(0, read=True, write=True)
        self.prog_fourier["tensorDimensions"] = dims
        self.prog_fourier.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # Step B.4: Green's function & spectral differentiation
        spectrum_tex.bind_to_image(0, read=True, write=False)
        self.grad_comp_x.bind_to_image(1, read=False, write=True)
        self.grad_comp_y.bind_to_image(2, read=False, write=True)
        self.prog_greens["tensorDimensions"] = dims
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size
        self.prog_greens.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # Step B.5: Inverse FFT for each force component
        real_x = self.run_fft(self.grad_comp_x, self.complex_tex_b, forward=False)
        real_y = self.run_fft(self.grad_comp_y, self.complex_tex_a, forward=False)

        # Step B.6: Pack force components
        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        self.gradient_tex.bind_to_image(3, read=False, write=True)
        self.prog_pack["tensorDimensions"] = dims
        self.prog_pack.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # Step C & D: Gather forces from all grids + Integration
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.offset_buf.bind_to_storage_buffer(2)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        self.prog_update["deltaTime"] = self.dt * self.timescale
        self.prog_update["tensorDimensions"] = dims
        self.prog_update["worldSize"] = self.world_size
        self.prog_update["voxelSize"] = self.voxel_size
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

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


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
