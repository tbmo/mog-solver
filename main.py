"""
Dimension Agnostic FFT Gravity
"""

import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import yaml


class Simulation(mglw.WindowConfig):
    gl_version = (4, 6)
    title = "FFT Gravity"
    window_size = (1024, 1024)
    aspect_ratio = 1.0  # <--- ADD THIS LINE BACK
    resizable = True  # You can keep this true if you want

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.key_event_func = self.key_event
        self.cfg = self.load_config()

        # --- NEW: Dimension Setup ---
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

    def load_config(self):
        # (Same as before)
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
        # Particles are vec4 (x,y,z,w). This works for 2D (z=0) and 3D.
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

    def init_textures(self):
        gs = self.grid_size
        # Dynamic tuple for dimensions: (gs, gs) or (gs, gs, gs)
        size = tuple([gs] * self.dim)

        # Helper to create correct texture type
        def create_tex(channels):
            if self.dim == 2:
                return self.ctx.texture(size, channels, dtype="f4")
            elif self.dim == 3:
                return self.ctx.texture3d(size, channels, dtype="f4")

        self.mass_tex = create_tex(1)
        self.complex_tex_a = create_tex(2)
        self.complex_tex_b = create_tex(2)
        self.gradient_tex = create_tex(4)  # stores grad.xyz + magnitude

        # Buffer size depends on total voxels
        total_voxels = gs**self.dim
        self.mass_buf = self.ctx.buffer(reserve=total_voxels * 4)

    def init_shaders(self):
        shader_dir = Path(__file__).parent / "shaders"

        # --- NEW: Inject Dimension Macro ---
        def load(name):
            src = (shader_dir / name).read_text()
            # Inject definition after #version
            header = f"#version 460\n#define DIM {self.dim}\n"
            # Remove original #version from file if present to avoid errors
            src = src.replace("#version 460", "")
            return header + src

        self.prog_mass = self.ctx.compute_shader(load("mass.glsl"))
        self.prog_complex = self.ctx.compute_shader(load("complex.glsl"))
        self.prog_fft = self.ctx.compute_shader(load("fft.glsl"))
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))
        self.prog_normalize = self.ctx.compute_shader(load("normalize.glsl"))
        self.prog_gradient = self.ctx.compute_shader(load("gradient.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))

    def init_particles(self):
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        # Randomize based on dimensions
        for i in range(self.dim):
            pos[:, i] = np.random.uniform(0, self.world_size, self.num_particles)
            vel[:, i] = np.random.uniform(-1.0, 1.0, self.num_particles)

        # Mass
        pos[:, 3] = self.cfg["particle_mass"]

        self.pos_buf.write(pos.tobytes())
        self.vel_buf.write(vel.tobytes())

    def init_render(self):
        # Basic perspective for 3D, ortho for 2D
        # (Simplified for brevity - assumes you want to see a slice or projection)
        self.render_prog = self.ctx.program(
            vertex_shader="""
            #version 460
            layout(location = 0) in vec4 in_pos;
            uniform float world_size;
            uniform int dim;
            
            void main() {
                // Normalize 0..world_size to -1..1
                vec3 p = (in_pos.xyz / world_size) * 2.0 - 1.0;
                
                if (dim == 3) {
                    // Simple perspective projection trick
                    float z = p.z + 2.0; // move camera back
                    gl_Position = vec4(p.x, p.y, 0.0, z); 
                    gl_PointSize = 4.0 / z; // depth scaling
                } else {
                    gl_Position = vec4(p.xy, 0.0, 1.0);
                    gl_PointSize = 2.0;
                }
            }
            """,
            fragment_shader="""
            #version 460
            out vec4 fragColor;
            void main() { fragColor = vec4(1.0, 0.8, 0.5, 1.0); }
            """,
        )
        self.render_prog["world_size"] = self.world_size
        self.render_prog["dim"] = self.dim
        self.vao = self.ctx.vertex_array(
            self.render_prog, [(self.pos_buf, "4f", "in_pos")]
        )

    def clear_mass(self):
        total_voxels = self.grid_size**self.dim
        zeros = np.zeros(total_voxels, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft(self, forward=True):
        gs = self.grid_size
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        src, dst = self.complex_tex_a, self.complex_tex_b

        # This works for 2, 3, (or 4 conceptually) dimensions automatically
        for axis in range(self.dim):
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis

            # Setup tuple for uniform: (gs, gs, 1) or (gs, gs, gs)
            dims = tuple([gs] * self.dim) + ((1,) if self.dim == 2 else ())
            self.prog_fft["tensorDimensions"] = dims

            for stage in range(num_stages + 1):
                self.prog_fft["stage"] = stage
                src.bind_to_image(0, read=True, write=False)
                dst.bind_to_image(1, read=False, write=True)

                # Dispatch
                # 2D: (gs/8, gs/8, 1)
                # 3D: (gs/8, gs/8, gs)
                z_groups = gs if self.dim == 3 else 1
                self.prog_fft.run(gs // 8, gs // 8, z_groups)

                self.ctx.memory_barrier()
                src, dst = dst, src

        # Ensure result ends in tex_a
        if src != self.complex_tex_a:
            # For strictness, you'd copy. For this demo, we just swap refs in step().
            pass

    def step(self):
        gs = self.grid_size
        dims = tuple([gs] * self.dim) + ((1,) if self.dim == 2 else ())
        n_particles = self.num_particles

        # Helper for dispatching 2D vs 3D grids
        def dispatch_grid():
            z = gs if self.dim == 3 else 1
            return (gs // 8, gs // 8, z)

        # 1. Clear Mass
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

        # 3. FFT
        self.run_fft(forward=True)

        # 4. Greens
        self.complex_tex_a.bind_to_image(0, read=True, write=False)
        self.complex_tex_b.bind_to_image(1, read=False, write=True)
        self.prog_greens["tensorDimensions"] = dims
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size
        self.prog_greens.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # Swap
        self.complex_tex_a, self.complex_tex_b = self.complex_tex_b, self.complex_tex_a

        # 5. IFFT
        self.run_fft(forward=False)

        # 6. Normalize
        self.complex_tex_a.bind_to_image(0, read=True, write=True)
        self.prog_normalize["tensorDimensions"] = dims
        self.prog_normalize.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 7. Gradient
        self.complex_tex_a.bind_to_image(0, read=True, write=False)
        self.gradient_tex.bind_to_image(1, read=False, write=True)
        self.prog_gradient["tensorDimensions"] = dims
        self.prog_gradient["voxelSize"] = self.voxel_size
        self.prog_gradient.run(*dispatch_grid())
        self.ctx.memory_barrier()

        # 8. Update Particles
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        self.prog_update["deltaTime"] = self.dt * self.timescale
        self.prog_update["tensorDimensions"] = dims
        self.prog_update["worldSize"] = self.world_size
        self.prog_update["voxelSize"] = self.voxel_size
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

    # ... key_event and on_render remain similar ...
    def on_render(self, time, frame_time):
        if not self.paused:
            self.step()

        self.ctx.clear(0.02, 0.02, 0.05)
        self.ctx.point_size = 2.0
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

            # --- NEW LOGIC START ---

            # Invert Time Direction
            elif key == keys.BACKSLASH:
                self.timescale *= -1.0
                print(f"Timescale Inverted: {self.timescale:.4f}")

            # Multiplicative Scaling (Exponential)
            # Use 1.5x steps for responsive zooming in/out of time
            elif key == keys.LEFT_BRACKET:  # [ -> Slow Down
                self.timescale /= 1.5
                print(f"Timescale: {self.timescale:.4f}")

            elif key == keys.RIGHT_BRACKET:  # ] -> Speed Up
                self.timescale *= 1.5
                print(f"Timescale: {self.timescale:.4f}")

            # --- NEW LOGIC END ---


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
