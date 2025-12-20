"""
Dimension Agnostic FFT Gravity
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import yaml


# --- Matrix Math Helpers (Numpy) ---
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

    # Create 4x4 view matrix
    view = np.identity(4, dtype="f4")
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[0, 3] = -np.dot(x_axis, eye)
    view[1, 3] = -np.dot(y_axis, eye)
    view[2, 3] = -np.dot(z_axis, eye)

    return view.T  # OpenGL expects column-major


class Simulation(mglw.WindowConfig):
    gl_version = (4, 6)
    title = "FFT Gravity"
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
        if self.dim > 3:
            raise NotImplementedError("OpenGL natively supports up to 3D textures.")

        self.timescale = 1.0
        self.paused = False

        self.grid_size = int(self.cfg["grid_size"])
        self.world_size = float(self.cfg["world_size"])
        self.voxel_size = self.world_size / self.grid_size
        self.num_particles = int(float(self.cfg["num_particles"]))
        self.G = self.cfg["G"]
        self.dt = self.cfg["dt"]

        self.init_buffers()
        self.init_textures()
        self.init_shaders()
        self.init_particles()
        self.init_render()

        # Camera State
        self.cam_rot_x = 0.0  # Azimuth
        self.cam_rot_y = 0.0  # Elevation
        self.cam_dist = 3.0  # Zoom radius
        self.mouse_pressed = False

    def load_config(self):
        cfg_path = Path(__file__).parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f)
        return {
            "grid_size": 128,
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

    def init_textures(self):
        gs = self.grid_size
        size = tuple([gs] * self.dim)

        def create_tex(channels):
            if self.dim == 2:
                return self.ctx.texture(size, channels, dtype="f4")
            elif self.dim == 3:
                return self.ctx.texture3d(size, channels, dtype="f4")

        self.mass_tex = create_tex(1)

        # FFT Ping-Pong textures
        self.complex_tex_a = create_tex(2)
        self.complex_tex_b = create_tex(2)
        self.complex_tex_c = create_tex(2)  # <--- NEW: Dedicated scratch for Z

        # Component Textures
        self.grad_comp_x = create_tex(2)
        self.grad_comp_y = create_tex(2)
        self.grad_comp_z = create_tex(2) if self.dim == 3 else None

        self.gradient_tex = create_tex(4)

        total_voxels = gs**self.dim
        self.mass_buf = self.ctx.buffer(reserve=total_voxels * 4)

    def mouse_drag_event(self, x, y, dx, dy):
        # Left Click Drag to Rotate
        if self.wnd.mouse_states.left:
            self.cam_rot_x -= dx * 0.5
            self.cam_rot_y += dy * 0.5

    def mouse_scroll_event(self, x_offset, y_offset):
        # Scroll to Zoom
        self.cam_dist -= y_offset * 0.2
        self.cam_dist = max(0.1, self.cam_dist)  # Clamp min zoom

    def init_shaders(self):
        shader_dir = Path(__file__).parent / "shaders"

        def load(name):
            src = (shader_dir / name).read_text()
            header = f"#version 460\n#define DIM {self.dim}\n"
            src = src.replace("#version 460", "")
            return header + src

        self.prog_mass = self.ctx.compute_shader(load("mass.glsl"))
        self.prog_complex = self.ctx.compute_shader(load("complex.glsl"))
        self.prog_fft = self.ctx.compute_shader(load("fft.glsl"))
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))

        self.prog_fourier = self.ctx.compute_shader(load("fourier.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))

        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))

    def init_particles(self):
        """Initialize particles using multi-octave Perlin noise with randomized parameters."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        # Randomize Perlin parameters
        num_octaves = np.random.randint(2, 6)
        base_freq = np.random.uniform(0.5, 4.0)
        lacunarity = np.random.uniform(1.5, 3.0)  # Frequency multiplier per octave
        persistence = np.random.uniform(0.3, 0.7)  # Amplitude multiplier per octave
        noise_strength = np.random.uniform(0.2, 0.5) * self.world_size

        # Random offset so each run looks different
        offset = np.random.uniform(0, 1000, size=self.dim)

        def perlin_octaves(coords):
            """Sample multi-octave Perlin noise at given coordinates."""
            total = np.zeros(len(coords))
            freq = base_freq
            amp = 1.0
            max_amp = 0.0

            for _ in range(num_octaves):
                # Perlin-like noise using sin of multiple frequencies
                noise = np.ones(len(coords))
                for d in range(coords.shape[1]):
                    noise *= np.sin(freq * coords[:, d] + offset[d])
                    noise += np.cos(freq * 1.7 * coords[:, d] + offset[d] * 0.7)
                total += noise * amp
                max_amp += amp
                freq *= lacunarity
                amp *= persistence

            return total / max_amp

        # Generate base positions - uniform random or grid with jitter
        if np.random.random() < 0.5:
            # Uniform random base
            base_pos = (
                np.random.uniform(0.1, 0.9, (self.num_particles, self.dim))
                * self.world_size
            )
        else:
            # Jittered grid base
            n_side = int(np.ceil(self.num_particles ** (1.0 / self.dim)))
            grid = np.meshgrid(
                *[np.linspace(0.1, 0.9, n_side) for _ in range(self.dim)]
            )
            base_pos = (
                np.stack([g.flatten() for g in grid], axis=1)[: self.num_particles]
                * self.world_size
            )
            base_pos += np.random.normal(0, self.world_size * 0.01, base_pos.shape)

        # Displace positions using noise
        for d in range(self.dim):
            # Use different slice of noise for each dimension
            noise_coords = base_pos / self.world_size * base_freq + offset[d] * 100
            displacement = perlin_octaves(noise_coords) * noise_strength
            base_pos[:, d] += displacement

        # Clamp to world bounds
        base_pos = np.clip(base_pos, 0.05 * self.world_size, 0.95 * self.world_size)

        pos[:, : self.dim] = base_pos
        pos[:, 3] = self.cfg["particle_mass"]

        # Initialize velocities - either zero, random, or curl noise
        # vel_mode = np.random.choice(["zero", "random", "curl"])
        vel_mode = "zero"

        if vel_mode == "zero":
            vel[:, : self.dim] = 0.0
        elif vel_mode == "random":
            speed = np.random.uniform(0.01, 0.1) * self.world_size
            vel[:, : self.dim] = np.random.normal(
                0, speed, (self.num_particles, self.dim)
            )
        elif vel_mode == "curl":
            # Curl noise - creates swirling patterns
            eps = 0.01 * self.world_size
            coords = pos[:, : self.dim] / self.world_size * base_freq

            if self.dim == 2:
                # 2D curl: (-dN/dy, dN/dx)
                n_py = perlin_octaves(coords + [0, eps])
                n_my = perlin_octaves(coords - [0, eps])
                n_px = perlin_octaves(coords + [eps, 0])
                n_mx = perlin_octaves(coords - [eps, 0])
                vel[:, 0] = -(n_py - n_my) / (2 * eps) * noise_strength * 0.5
                vel[:, 1] = (n_px - n_mx) / (2 * eps) * noise_strength * 0.5
            elif self.dim == 3:
                # 3D curl from 3 noise fields
                for d in range(3):
                    d1, d2 = (d + 1) % 3, (d + 2) % 3
                    offset1 = np.zeros(3)
                    offset1[d1] = eps
                    offset2 = np.zeros(3)
                    offset2[d2] = eps
                    n1p = perlin_octaves(coords + offset1 + d * 50)
                    n1m = perlin_octaves(coords - offset1 + d * 50)
                    n2p = perlin_octaves(coords + offset2 + d * 50)
                    n2m = perlin_octaves(coords - offset2 + d * 50)
                    vel[:, d] = (
                        ((n1p - n1m) - (n2p - n2m)) / (2 * eps) * noise_strength * 0.3
                    )

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
                    // 1. Normalize 0..world_size to -1..1 (Unit Cube)
                    vec3 p = (in_pos.xyz / world_size) * 2.0 - 1.0;

                    // 2. Apply Camera Matrices
                    gl_Position = m_proj * m_view * vec4(p, 1.0);
                    
                    // 3. Distance-based point sizing
                    float dist = gl_Position.w;
                    gl_PointSize = 600.0 / (dist + 0.1); // Adjust 600.0 for size preference
                }
                """,
            fragment_shader="""
                #version 460
                out vec4 fragColor;
                void main() { 
                    // Simple orange/white glow
                    fragColor = vec4(1.0, 1.0, 1.0, 1.0); 
                }
                """,
        )
        self.render_prog["world_size"] = self.world_size
        self.vao = self.ctx.vertex_array(
            self.render_prog, [(self.pos_buf, "4f", "in_pos")]
        )

    def clear_mass(self):
        total_voxels = self.grid_size**self.dim
        zeros = np.zeros(total_voxels, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft(self, src_tex, dst_tex, forward=True):
        gs = self.grid_size
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        curr_src = src_tex
        curr_dst = dst_tex

        for axis in range(self.dim):
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis
            dims = tuple([gs] * self.dim) + ((1,) if self.dim == 2 else ())
            self.prog_fft["tensorDimensions"] = dims

            for stage in range(num_stages + 1):
                self.prog_fft["stage"] = stage

                curr_src.bind_to_image(0, read=True, write=False)
                curr_dst.bind_to_image(1, read=False, write=True)

                z_groups = gs if self.dim == 3 else 1
                self.prog_fft.run(gs // 8, gs // 8, z_groups)
                self.ctx.memory_barrier()

                curr_src, curr_dst = curr_dst, curr_src

        return curr_src

    def step(self):
        gs = self.grid_size
        dims = tuple([gs] * self.dim) + ((1,) if self.dim == 2 else ())
        n_particles = self.num_particles

        def dispatch_grid():
            z = gs if self.dim == 3 else 1
            return (gs // 8, gs // 8, z)

        # # 0. Center particles on COM
        # pos_data = np.frombuffer(self.pos_buf.read(), dtype="f4").reshape(-1, 4).copy()
        # com = np.average(pos_data[:, : self.dim], axis=0, weights=pos_data[:, 3])
        # shift = self.world_size / 2.0 - com
        # pos_data[:, : self.dim] += shift
        # pos_data[:, : self.dim] %= self.world_size  # wrap periodic
        # self.pos_buf.write(pos_data.tobytes())

        # 1. Clear & Mass
        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.prog_mass["tensorDimensions"] = dims
        self.prog_mass["voxelSize"] = self.voxel_size
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # 2. Mass -> Complex
        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        self.prog_complex["tensorDimensions"] = dims
        self.prog_complex.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 3. Forward FFT
        spectrum_tex = self.run_fft(
            self.complex_tex_a, self.complex_tex_b, forward=True
        )

        # 3.5. Fourier space manipulation (zeroing DC for now)
        spectrum_tex.bind_to_image(0, read=True, write=True)
        self.prog_fourier["tensorDimensions"] = dims
        self.prog_fourier.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 4. Greens & Derivatives
        spectrum_tex.bind_to_image(0, read=True, write=False)
        self.grad_comp_x.bind_to_image(1, read=False, write=True)
        self.grad_comp_y.bind_to_image(2, read=False, write=True)

        self.prog_greens["tensorDimensions"] = dims
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size

        if self.dim == 3:
            self.grad_comp_z.bind_to_image(3, read=False, write=True)

        self.prog_greens.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 5. Inverse FFT - The Clean Pattern

        # X uses B as scratch
        real_x = self.run_fft(self.grad_comp_x, self.complex_tex_b, forward=False)

        # Y uses A as scratch
        real_y = self.run_fft(self.grad_comp_y, self.complex_tex_a, forward=False)

        real_z = None
        if self.dim == 3:
            # Z uses C as scratch (New Pattern)
            real_z = self.run_fft(self.grad_comp_z, self.complex_tex_c, forward=False)

        # 6. Pack
        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        if self.dim == 3:
            real_z.bind_to_image(2, read=True, write=False)

        self.gradient_tex.bind_to_image(3, read=False, write=True)
        self.prog_pack["tensorDimensions"] = dims
        self.prog_pack.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 7. Update Particles
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
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
        self.ctx.enable(moderngl.BLEND)  # Optional: nice for overlapping particles

        # --- CAMERA MATH ---
        # 1. Calculate Eye Position from spherical coordinates
        # rot_x = azimuth, rot_y = elevation
        rad_x = np.radians(self.cam_rot_x)
        rad_y = np.radians(np.clip(self.cam_rot_y, -89, 89))

        eye_x = self.cam_dist * np.sin(rad_x) * np.cos(rad_y)
        eye_y = self.cam_dist * np.sin(rad_y)
        eye_z = self.cam_dist * np.cos(rad_x) * np.cos(rad_y)

        eye = np.array([eye_x, eye_y, eye_z], dtype="f4")
        target = np.array([0, 0, 0], dtype="f4")  # Look at center of box
        up = np.array([0, 1, 0], dtype="f4")

        # 2. Generate Matrices
        m_view = create_look_at(eye, target, up)
        m_proj = create_perspective_projection(45.0, self.aspect_ratio, 0.1, 100.0)

        # 3. Upload to Shader
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
