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

        self.prog_normalize = self.ctx.compute_shader(load("normalize.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))

        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))

    def init_particles(self):
        """Initialize particles in planetary orbital configuration."""
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        center = self.world_size / 2.0

        # Central massive body (or cluster of heavy particles)
        num_central = max(1, self.num_particles // 100)  # 1% as central mass
        central_mass = self.cfg["particle_mass"] * 100  # Much heavier

        for i in range(num_central):
            pos[i, : self.dim] = center + np.random.normal(
                0, self.world_size * 0.01, self.dim
            )
            vel[i, : self.dim] = 0.0  # Central mass stationary
            pos[i, 3] = central_mass

        # Orbiting particles
        for i in range(num_central, self.num_particles):
            # Random orbital radius (weighted toward middle distances)
            r = np.random.uniform(0.1, 0.45) * self.world_size

            if self.dim == 2:
                # 2D: circular orbits in XY plane
                theta = np.random.uniform(0, 2 * np.pi)
                pos[i, 0] = center + r * np.cos(theta)
                pos[i, 1] = center + r * np.sin(theta)

                # Circular orbital velocity: v = sqrt(GM/r)
                total_central_mass = num_central * central_mass
                v_orbital = np.sqrt(self.G * total_central_mass / r)

                # Tangential velocity (perpendicular to radius)
                vel[i, 0] = -v_orbital * np.sin(theta)
                vel[i, 1] = v_orbital * np.cos(theta)

            elif self.dim == 3:
                # 3D: disk-like distribution with some thickness
                theta = np.random.uniform(0, 2 * np.pi)
                # Slight inclination for thickness
                phi = np.random.normal(np.pi / 2, 0.1)  # Mostly in XZ plane

                pos[i, 0] = center + r * np.sin(phi) * np.cos(theta)
                pos[i, 1] = center + r * np.cos(phi)  # Y is "up"
                pos[i, 2] = center + r * np.sin(phi) * np.sin(theta)

                # Orbital velocity in XZ plane
                total_central_mass = num_central * central_mass
                v_orbital = np.sqrt(self.G * total_central_mass / r)

                # Tangent vector in XZ plane (perpendicular to radial)
                vel[i, 0] = -v_orbital * np.sin(theta)
                vel[i, 1] = 0.0
                vel[i, 2] = v_orbital * np.cos(theta)

            # Add small random perturbation for realism
            vel[i, : self.dim] += np.random.normal(0, v_orbital * 0.05, self.dim)
            pos[i, 3] = self.cfg["particle_mass"]

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
                    fragColor = vec4(1.0, 0.9, 0.7, 1.0); 
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
