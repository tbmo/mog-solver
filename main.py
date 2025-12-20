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
        self.offsets, self.rotations = self.compute_transforms()

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

        effective_res = self.grid_size * self.n_grids
        total_cells = self.n_grids * (self.grid_size**self.dim)
        equiv_cells = effective_res**self.dim

        print(f"Linear-Sparse-Grid Config ({self.dim}D):")
        print(f"  world_size: {self.world_size}")
        # particle count
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
        phi = (1 + np.sqrt(5)) / 2  # golden ratio

        # The 12 vertices of an icosahedron (normalized)
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

        # Identity
        add_rotation(np.eye(3))

        # 5-fold axes through vertices: 72°, 144°, 216°, 288°
        for v in icosa_verts:
            for k in [1, 2, 3, 4]:
                angle = k * 2 * np.pi / 5
                add_rotation(rotation_matrix(v, angle))

        # 3-fold axes through face centers: 120°, 240°
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

        # 2-fold axes through edge midpoints: 180°
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

        # Should be exactly 60
        assert len(rotations) == 60, f"Expected 60 rotations, got {len(rotations)}"

        return rotations

    def get_octahedral_rotations(self):
        """Returns all 24 proper rotations of a cube as 3x3 matrices."""
        rotations = []

        # All permutations of (x,y,z)
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

        # All sign combinations
        signs = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]

        for perm in perms:
            for sign in signs:
                mat = np.zeros((3, 3), dtype="f4")
                for i in range(3):
                    mat[i, perm[i]] = sign[i]

                # Only keep proper rotations (det = +1)
                if np.linalg.det(mat) > 0:
                    rotations.append(mat)

        return rotations

    def get_dihedral_rotations(self):
        """Returns all 8 rotations/reflections of a square as 2x2 matrices (D4 group)."""
        rotations = []

        # 4 rotations (0, 90, 180, 270 degrees)
        for k in range(4):
            angle = k * np.pi / 2
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype="f4")
            rotations.append(rot)

        # 4 reflections (across x, y, and two diagonals)
        # Reflection across x-axis then rotate
        reflect = np.array([[1, 0], [0, -1]], dtype="f4")
        for k in range(4):
            angle = k * np.pi / 2
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype="f4")
            rotations.append(rot @ reflect)

        return rotations

    # def compute_transforms(self):
    #     """
    #     Compute offset + rotation for each grid.
    #     Uses tetrahedral offsets + octahedral rotations for 3D.
    #     Uses diagonal offsets + dihedral rotations for 2D.
    #     """
    #     offsets = np.zeros((self.n_grids, 4), dtype="f4")
    #     # mat3 packed as 3 vec4s (12 floats) for 3D, mat2 packed as 2 vec2s (4 floats) for 2D
    #     # Use 12 floats for both for simplicity (2D just uses top-left 2x2)
    #     rotations = np.zeros((self.n_grids, 12), dtype="f4")

    #     if self.dim == 2:
    #         dihedral = self.get_dihedral_rotations()  # 8 rotations

    #         # Square diagonal directions
    #         diag_dirs = np.array(
    #             [
    #                 [1, 1],
    #                 [1, -1],
    #                 [-1, 1],
    #                 [-1, -1],
    #             ],
    #             dtype="f4",
    #         ) / np.sqrt(2)

    #         for i in range(self.n_grids):
    #             frac = i / self.n_grids

    #             # Offset: diagonal direction
    #             d = diag_dirs[i % 4]
    #             offsets[i, 0:2] = frac * self.voxel_size * d

    #             # Rotation: cycle through dihedral group
    #             rot = dihedral[i % 8]
    #             # Pack mat2 into first 4 floats (column-major for GLSL)
    #             rotations[i, 0:2] = rot[:, 0]  # first column
    #             rotations[i, 4:6] = rot[
    #                 :, 1
    #             ]  # second column (offset by 4 for vec4 alignment)

    #     else:  # 3D
    #         octahedral = self.get_octahedral_rotations()  # 24 rotations

    #         # Tetrahedral offset directions (maximally symmetric)
    #         tetra_dirs = np.array(
    #             [
    #                 [1, 1, 1],
    #                 [1, -1, -1],
    #                 [-1, 1, -1],
    #                 [-1, -1, 1],
    #             ],
    #             dtype="f4",
    #         ) / np.sqrt(3)

    #         for i in range(self.n_grids):
    #             frac = i / self.n_grids

    #             # Offset: tetrahedral direction
    #             d = tetra_dirs[i % 4]
    #             offsets[i, 0:3] = frac * self.voxel_size * d

    #             # Rotation: cycle through octahedral group
    #             rot = octahedral[i % 24]
    #             # Pack mat3 as 3 vec4s (column-major for GLSL)
    #             rotations[i, 0:3] = rot[:, 0]  # first column
    #             rotations[i, 4:7] = rot[:, 1]  # second column
    #             rotations[i, 8:11] = rot[:, 2]  # third column

    #     return offsets, rotations

    def compute_transforms(self):
        offsets = np.zeros((self.n_grids, 4), dtype="f4")
        rotations = np.zeros((self.n_grids, 12), dtype="f4")

        if self.dim == 2:
            # Keep dihedral for 2D
            dihedral = self.get_dihedral_rotations()
            diag_dirs = np.array(
                [
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1],
                ],
                dtype="f4",
            ) / np.sqrt(2)

            for i in range(self.n_grids):
                frac = i / self.n_grids
                d = diag_dirs[i % 4]
                offsets[i, 0:2] = frac * self.voxel_size * d
                rot = dihedral[i % 8]
                rotations[i, 0:2] = rot[:, 0]
                rotations[i, 4:6] = rot[:, 1]

        else:  # 3D
            icosahedral = self.get_icosahedral_rotations()  # 60 rotations

            # Golden ratio offset directions - 20 vertices of dodecahedron
            # (dual to icosahedron, same symmetry group)
            phi = (1 + np.sqrt(5)) / 2
            dodeca_verts = np.array(
                [
                    [1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, -1],
                    [0, phi, 1 / phi],
                    [0, phi, -1 / phi],
                    [0, -phi, 1 / phi],
                    [0, -phi, -1 / phi],
                    [1 / phi, 0, phi],
                    [1 / phi, 0, -phi],
                    [-1 / phi, 0, phi],
                    [-1 / phi, 0, -phi],
                    [phi, 1 / phi, 0],
                    [phi, -1 / phi, 0],
                    [-phi, 1 / phi, 0],
                    [-phi, -1 / phi, 0],
                ],
                dtype="f4",
            )
            dodeca_verts /= np.linalg.norm(dodeca_verts[0])

            for i in range(self.n_grids):
                frac = i / self.n_grids

                # Offset: cycle through dodecahedron vertices
                d = dodeca_verts[i % 20]
                offsets[i, 0:3] = frac * self.voxel_size * d

                # Rotation: cycle through icosahedral group
                rot = icosahedral[i % 60]
                rotations[i, 0:3] = rot[:, 0]
                rotations[i, 4:7] = rot[:, 1]
                rotations[i, 8:11] = rot[:, 2]

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
        # Particle buffers
        self.pos_buf = self.ctx.buffer(reserve=self.num_particles * 16)
        self.vel_buf = self.ctx.buffer(reserve=self.num_particles * 16)

        # Offset buffer for GPU access (vec4 aligned)
        self.offset_buf = self.ctx.buffer(self.offsets.tobytes())
        self.rotation_buf = self.ctx.buffer(self.rotations.tobytes())

        # Mass buffer: one contiguous buffer for all grids
        # Layout: [grid0_cells..., grid1_cells..., ...]
        cells_per_grid = self.grid_size**self.dim
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
                return [
                    self.ctx.texture3d(size, channels, dtype="f4") for _ in range(ng)
                ]

            self.complex_tex_a = create_tex_array(2)
            self.complex_tex_b = create_tex_array(2)
            self.complex_tex_c = create_tex_array(2)

            self.grad_comp_x = create_tex_array(2)
            self.grad_comp_y = create_tex_array(2)
            self.grad_comp_z = create_tex_array(2)

            self.gradient_tex = create_tex_array(4)

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
        self.prog_greens = self.ctx.compute_shader(load("greens.glsl"))
        self.prog_fourier = self.ctx.compute_shader(load("fourier.glsl"))
        self.prog_pack = self.ctx.compute_shader(load("pack.glsl"))
        self.prog_update = self.ctx.compute_shader(load("update.glsl"))

    def init_particles(self):
        """Initialize particles in a simple centered cluster."""
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

            # Extract rotation matrix from packed format
            rot_data = self.rotations[grid_idx]

            if self.dim == 2:
                # Unpack mat2 (column-major): col0 at [0:2], col1 at [4:6]
                rot = np.array(
                    [[rot_data[0], rot_data[4]], [rot_data[1], rot_data[5]]], dtype="f4"
                )

                # Define unit cell corners centered at origin
                corners = np.array(
                    [
                        [0, 0],
                        [vs, 0],
                        [vs, vs],
                        [0, vs],
                    ],
                    dtype="f4",
                )

                # Rotate corners around cell center, then translate
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

                # Draw square outline (4 edges)
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
                # Unpack mat3 (column-major): col0 at [0:3], col1 at [4:7], col2 at [8:11]
                rot = np.array(
                    [
                        [rot_data[0], rot_data[4], rot_data[8]],
                        [rot_data[1], rot_data[5], rot_data[9]],
                        [rot_data[2], rot_data[6], rot_data[10]],
                    ],
                    dtype="f4",
                )

                # Define unit cube corners centered at origin
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

                # Rotate corners around cell center, then translate
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

                # 12 edges of a cube using corner indices
                # Corners: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(1,1,0),
                #          4=(0,0,1), 5=(1,0,1), 6=(0,1,1), 7=(1,1,1)
                edges = [
                    (0, 1),
                    (0, 2),
                    (0, 4),  # from corner 0
                    (1, 3),
                    (1, 5),  # from corner 1
                    (2, 3),
                    (2, 6),  # from corner 2
                    (3, 7),  # from corner 3
                    (4, 5),
                    (4, 6),  # from corner 4
                    (5, 7),
                    (6, 7),  # from corners 5, 6
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
        self.rotation_buf.bind_to_storage_buffer(3)
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
        spectrum_tex = self.run_fft_2d(
            self.complex_tex_a, self.complex_tex_b, forward=True
        )

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
        self.rotation_buf.bind_to_storage_buffer(3)
        set_uniform(self.prog_mass, "gridSize", gs)
        set_uniform(self.prog_mass, "nGrids", ng)
        set_uniform(self.prog_mass, "voxelSize", self.voxel_size)
        set_uniform(self.prog_mass, "worldSize", self.world_size)
        self.prog_mass.run((n_particles + 255) // 256)
        self.ctx.memory_barrier()

        # Process each grid independently for FFT to avoid spectral leakage

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
                forward=True,
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
                forward=False,
            )
            real_y = self.run_fft_3d_single(
                self.grad_comp_y[grid_idx],
                self.complex_tex_a[grid_idx],
                self.complex_tex_c[grid_idx],
                forward=False,
            )
            real_z = self.run_fft_3d_single(
                self.grad_comp_z[grid_idx],
                self.complex_tex_c[grid_idx],
                self.complex_tex_b[grid_idx],
                forward=False,
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

    def on_key_event(self, key, action, modifiers):
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

            elif key == keys.G:
                self.show_grid_debug = not self.show_grid_debug
                print(f"Grid debug: {'ON' if self.show_grid_debug else 'OFF'}")


if __name__ == "__main__":
    mglw.run_window_config(Simulation, args=["--window", "glfw"])
