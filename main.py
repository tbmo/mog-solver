"""
Minimal 2D FFT Gravity Simulation
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
    aspect_ratio = 1.0
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FORCE the window to use this class's key_event handler
        self.wnd.key_event_func = self.key_event

        self.cfg = self.load_config()
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
        cfg_path = Path(__file__).parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f)
        # defaults
        return {
            "grid_size": 128,
            "world_size": 100.0,
            "num_particles": 4096,
            "G": 1.0,
            "dt": 0.016,
            "particle_mass": 1.0,
        }

    def init_buffers(self):
        # Particle positions (x, y, z, mass) - z=0 for 2D
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        # Particle velocities (vx, vy, vz, unused)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

    def init_textures(self):
        gs = self.grid_size
        # Mass texture (r32ui) - but moderngl doesn't support r32ui well, use r32f
        self.mass_tex = self.ctx.texture((gs, gs), 1, dtype="f4")
        # Complex textures for FFT ping-pong (rg32f)
        self.complex_tex_a = self.ctx.texture((gs, gs), 2, dtype="f4")
        self.complex_tex_b = self.ctx.texture((gs, gs), 2, dtype="f4")
        # Gradient texture (rgba32f) - gradient.xy + magnitude
        self.gradient_tex = self.ctx.texture((gs, gs), 4, dtype="f4")

        # For atomic mass accumulation, we need a buffer instead
        self.mass_buf = self.ctx.buffer(reserve=gs * gs * 4)

    def init_shaders(self):
        shader_dir = Path(__file__).parent / "shaders"

        def load(name):
            return (shader_dir / name).read_text()

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

        # Uniform distribution across the whole world_size
        # x, y coordinates
        pos[:, 0] = np.random.uniform(0, self.world_size, self.num_particles)
        pos[:, 1] = np.random.uniform(0, self.world_size, self.num_particles)
        pos[:, 2] = 0  # z = 0 for 2D
        pos[:, 3] = self.cfg["particle_mass"]  # mass

        # Random velocities (thermal noise)
        # Small values so they drift rather than orbit immediately
        vel[:, 0] = np.random.uniform(-1.0, 1.0, self.num_particles)
        vel[:, 1] = np.random.uniform(-1.0, 1.0, self.num_particles)

        self.pos_buf.write(pos.tobytes())
        self.vel_buf.write(vel.tobytes())

    def init_render(self):
        # Simple point rendering
        self.render_prog = self.ctx.program(
            vertex_shader="""
            #version 460
            layout(location = 0) in vec4 in_pos;
            uniform float world_size;
            void main() {
                vec2 ndc = (in_pos.xy / world_size) * 2.0 - 1.0;
                gl_Position = vec4(ndc, 0.0, 1.0);
                gl_PointSize = 2.0;
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
        # Zero out mass buffer (uint)
        zeros = np.zeros(self.grid_size * self.grid_size, dtype="u4")
        self.mass_buf.write(zeros.tobytes())

    def run_fft(self, forward=True):
        """Run 2D FFT on complex_tex_a, result in complex_tex_a"""
        gs = self.grid_size
        num_stages = int(np.log2(gs))
        direction = 1 if forward else -1

        src, dst = self.complex_tex_a, self.complex_tex_b

        for axis in range(2):  # X then Y
            self.prog_fft["direction"] = direction
            self.prog_fft["axis"] = axis
            self.prog_fft["tensorDimensions"] = (gs, gs, 1)

            for stage in range(num_stages + 1):  # 0 = bit reversal, 1..n = butterflies
                self.prog_fft["stage"] = stage

                src.bind_to_image(0, read=True, write=False)
                dst.bind_to_image(1, read=False, write=True)

                self.prog_fft.run(gs // 8, gs // 8, 1)
                self.ctx.memory_barrier()

                src, dst = dst, src

        # Result is in src after all swaps
        if src != self.complex_tex_a:
            # Copy back if needed (or just track which is "current")
            pass

    def step(self):
        gs = self.grid_size
        n_particles = self.num_particles

        # 1. Clear and deposit mass
        self.clear_mass()
        self.pos_buf.bind_to_storage_buffer(0)
        self.mass_buf.bind_to_storage_buffer(1)
        self.prog_mass["tensorDimensions"] = (gs, gs, 1)
        self.prog_mass["voxelSize"] = self.voxel_size
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # 2. Mass buffer -> complex texture
        self.mass_buf.bind_to_storage_buffer(0)
        self.complex_tex_a.bind_to_image(1, read=False, write=True)
        self.prog_complex["tensorDimensions"] = (gs, gs, 1)
        self.prog_complex.run(gs // 8, gs // 8, 1)
        self.ctx.memory_barrier()

        # 3. Forward FFT
        self.run_fft(forward=True)

        # 4. Green's function (multiply by -4πG/k²)
        self.complex_tex_a.bind_to_image(0, read=True, write=False)
        self.complex_tex_b.bind_to_image(1, read=False, write=True)
        self.prog_greens["tensorDimensions"] = (gs, gs, 1)
        self.prog_greens["G"] = self.G
        self.prog_greens["worldSize"] = self.world_size
        self.prog_greens.run(gs // 8, gs // 8, 1)
        self.ctx.memory_barrier()

        # Swap so potential is in tex_a
        self.complex_tex_a, self.complex_tex_b = self.complex_tex_b, self.complex_tex_a

        # 5. Inverse FFT
        self.run_fft(forward=False)

        # 6. Normalize
        self.complex_tex_a.bind_to_image(0, read=True, write=True)
        self.prog_normalize["tensorDimensions"] = (gs, gs, 1)
        self.prog_normalize.run(gs // 8, gs // 8, 1)
        self.ctx.memory_barrier()

        # 7. Gradient
        self.complex_tex_a.bind_to_image(0, read=True, write=False)
        self.gradient_tex.bind_to_image(1, read=False, write=True)
        self.prog_gradient["tensorDimensions"] = (gs, gs, 1)
        self.prog_gradient["voxelSize"] = self.voxel_size
        self.prog_gradient.run(gs // 8, gs // 8, 1)
        self.ctx.memory_barrier()

        # 8. Update particles
        self.pos_buf.bind_to_storage_buffer(0)
        self.vel_buf.bind_to_storage_buffer(1)
        self.gradient_tex.bind_to_image(0, read=True, write=False)
        self.prog_update["deltaTime"] = self.dt * self.timescale
        self.prog_update["tensorDimensions"] = (gs, gs, 1)
        self.prog_update["worldSize"] = self.world_size
        self.prog_update["voxelSize"] = self.voxel_size
        self.prog_update.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

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
