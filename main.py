import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import yaml
import subprocess
import time
from scipy.stats import qmc


class VideoRecorder:
    def __init__(self, width, height, fps=60, output="mog_simulation.mp4"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output = output
        self.proc = None
        self.pbo = None
        self.pbo_index = 0
        self._pbo_ready = False
        self._queue = None
        self._thread = None

    def start(self, ctx):  # <-- ctx passed in here now
        import queue, threading

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            self.output,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        # Create two PBOs for ping-pong async readback
        self.pbo = [ctx.buffer(reserve=self.width * self.height * 3) for _ in range(2)]
        self.pbo_index = 0
        self._pbo_ready = False

        # Background writer thread so stdin.write() never blocks render
        self._queue = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

        print(f"Recording started -> {self.output}")

    def _writer(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            self.proc.stdin.write(item)

    def write_frame(self, ctx, wnd):
        w, h = wnd.size
        cur = self.pbo_index
        nxt = 1 - cur

        # Read the frame that was downloaded last call (no GPU stall)
        if self._pbo_ready:
            data = self.pbo[nxt].read()
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[::-1].tobytes()
            try:
                self._queue.put_nowait(arr)
            except Exception:
                pass  # drop frame rather than stall

        # Kick off async download of current frame into cur PBO
        ctx.screen.read_into(self.pbo[cur], viewport=(0, 0, w, h), components=3)
        self.pbo_index = nxt
        self._pbo_ready = True

    def stop(self):
        if self._thread:
            self._queue.put(None)
            self._thread.join()
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
            print("Recording saved.")


class RandomSE3Sampler:
    def __init__(self, grid_size, world_size, n_grids, seed=None):
        self.grid_size = grid_size
        self.world_size = world_size
        self.voxel_size = world_size / grid_size
        self.n_grids = n_grids
        self.seed = seed

    def generate(self):
        rng = np.random.default_rng(self.seed)
        n = self.n_grids

        u = rng.uniform(0, 1, (n, 3))
        q0 = np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1])
        q1 = np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1])
        q2 = np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2])
        q3 = np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])
        quats = np.stack([q0, q1, q2, q3], axis=1).astype(np.float32)

        offsets = rng.uniform(0, self.voxel_size, (n, 3)).astype(np.float32)
        rotations = np.array([self._quat_to_matrix(q) for q in quats])
        return rotations, offsets

    def get_transforms(self, verbose=False):
        if verbose:
            print(f"  Generating {self.n_grids} random SE(3) transforms...")
        return self.generate()

    def _quat_to_matrix(self, q):
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                    0.0,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                    0.0,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ).T[:3, :4]


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
    window_size = (1024, 768)
    aspect_ratio = window_size[0] / window_size[1]
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
        total_cells = self.n_grids * (self.grid_size**3)
        equiv_cells = effective_res**3

        solver_mode = "Direct Convolution" if self.use_convolution else "FFT"
        print("Linear-Sparse-Grid Config (3D):")
        print(f"  Solver: {solver_mode}")
        print(f"  world_size: {self.world_size}")
        print(f"  num_particles: {self.num_particles:,}")
        print(f"  n_grids: {self.n_grids}")
        print(f"  grid_size: {self.grid_size}^3")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Equivalent single grid: {effective_res}^3 = {equiv_cells:,} cells")
        print(f"  Memory savings: {equiv_cells / total_cells:.1f}x")

        # Recording
        self.recorder = None
        self.recording = False
        if self.cfg.get("record", False):
            w, h = self.wnd.size
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output = self.cfg.get("record_output", f"mog_{timestamp}.mp4")
            fps = int(self.cfg.get("record_fps", 60))
            self.recorder = VideoRecorder(w, h, fps=fps, output=output)
            self.recorder.start(self.ctx)
            self.recording = True

    def compute_transforms(self):
        offsets = np.zeros((self.n_grids, 4), dtype="f4")
        rotations = np.zeros((self.n_grids, 12), dtype="f4")

        sampler = RandomSE3Sampler(
            grid_size=self.grid_size, world_size=self.world_size, n_grids=self.n_grids
        )
        rot_mats, offs = sampler.get_transforms()

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
            "solver": "fft",
            "record": True,
        }

    def init_buffers(self):
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())
        self.rotation_buf = self.ctx.buffer(self.rotations.tobytes())

        cells_per_grid = self.grid_size**3
        total_cells = self.n_grids * cells_per_grid
        self.mass_buf = self.ctx.buffer(reserve=total_cells * 4)

        w, h = self.wnd.size
        self.pbo = [self.ctx.buffer(reserve=w * h * 3) for _ in range(2)]
        self.pbo_index = 0

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

    def init_greens_kernel(self):
        gs = self.grid_size
        voxel = self.voxel_size

        kernel = np.zeros((gs, gs, gs, 4), dtype="f4")

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
                        factor = 1.0 / (4.0 * np.pi * r2 * r)
                        kernel[iz, iy, ix, 0] = rx * factor
                        kernel[iz, iy, ix, 1] = ry * factor
                        kernel[iz, iy, ix, 2] = rz * factor

        print(f"Green's kernel (3D): {gs}x{gs}x{gs}x4, voxel={voxel:.2f}")

        self.green_kernel_buf = self.ctx.buffer(kernel.tobytes())
        print(f"  Kernel buffer size: {kernel.nbytes / 1024:.1f} KB")

    def init_particles(self):
        self.init_particles_gpu(noise_scale=4.0, density_contrast=2.0, seed=12345)

    def init_particles_gpu(
        self, noise_scale=4.0, density_contrast=2.0, seed=None, spawn_buffer=None
    ):
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
        self.prog_init_particles["spawnShape"] = 1

        self.prog_init_particles.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        print(
            f"GPU init: {n_particles:,} particles, scale={noise_scale}, contrast={density_contrast}, buffer={spawn_buffer:.0%}"
        )

    def init_particles2(self):
        pos = np.zeros((self.num_particles, 4), dtype="f4")
        vel = np.zeros((self.num_particles, 4), dtype="f4")

        center = self.world_size / 2.0
        n_per_galaxy = self.num_particles // 2

        for i, (offset, v_bulk) in enumerate(
            [
                (np.array([-0.2, 0.0, 0.0]), np.array([0.3, 0.1, 0.0])),
                (np.array([0.2, 0.0, 0.0]), np.array([-0.3, -0.1, 0.0])),
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
                    vec2 coord = gl_PointCoord * 2.0 - 1.0;
                    float r = dot(coord, coord);
                    if (r > 1.0) discard;
                    float alpha = exp(-r * 3.0);
                    fragColor = vec4(1.0, 1.0, 1.0, alpha);
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

    def init_grid_debug_geometry(self, cells_per_axis=8, max_grids=1):
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

            rot = np.array(
                [
                    [rot_data[0], rot_data[4], rot_data[8]],
                    [rot_data[1], rot_data[5], rot_data[9]],
                    [rot_data[2], rot_data[6], rot_data[10]],
                ],
                dtype="f4",
            )

            world_offset = np.array(
                [center + offset[0], center + offset[1], center + offset[2]], dtype="f4"
            )

            for cx in range(-half_extent, half_extent + 1):
                for cy in range(-half_extent, half_extent + 1):
                    for cz in range(-half_extent, half_extent + 1):
                        cell_origin = np.array([cx * vs, cy * vs, cz * vs], dtype="f4")

                        corners = np.array(
                            [
                                cell_origin + [0, 0, 0],
                                cell_origin + [vs, 0, 0],
                                cell_origin + [0, vs, 0],
                                cell_origin + [vs, vs, 0],
                                cell_origin + [0, 0, vs],
                                cell_origin + [vs, 0, vs],
                                cell_origin + [0, vs, vs],
                                cell_origin + [vs, vs, vs],
                            ],
                            dtype="f4",
                        )

                        rotated_corners = []
                        for c in corners:
                            rotated = rot @ c
                            world = rotated + world_offset
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
                                [list(rotated_corners[i]), list(rotated_corners[j])]
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
        cells_per_grid = self.grid_size**3
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
        self.step_fft()

    def on_render(self, time, frame_time):
        if not self.paused:
            self.step()

        if self.show_grid_debug:
            self.init_grid_debug_geometry()

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

        if self.recording and self.recorder:
            self.recorder.write_frame(self.ctx, self.wnd)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.Q:
                if self.recording and self.recorder:
                    self.recorder.stop()
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

            elif key == keys.F:
                from force_probe import probe_force_law_v2

                probe_force_law_v2(self)


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
