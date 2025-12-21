import numpy as np


class HierarchicalSE3Sampler:
    """
    Jointly optimizes rotation + translation for grid sampling.

    The configuration space is SO(3) × T³ where T³ is the 3-torus of
    translations mod voxel_size. We want each (rotation, offset) pair
    to create a maximally different grid tessellation of space.
    """

    def __init__(self, grid_size, world_size, n_grids, seed=42):
        self.grid_size = grid_size
        self.world_size = world_size
        self.voxel_size = world_size / grid_size
        self.n_grids = n_grids
        self.rng = np.random.default_rng(seed)
        self._configs = []
        self._max_computed = 0

        # Auto-tune weights based on geometry
        self.w_rot, self.w_trans = self._compute_weights()

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

        # Characteristic length where rotation has effect
        # Points near center don't move much under rotation
        # Points at edge move a lot
        # Use RMS distance from center: W/(2√3) for uniform cube
        char_radius = W / (2 * np.sqrt(3))

        # Maximum displacement from rotation (full π rotation at char_radius)
        max_rot_displacement = char_radius * np.pi

        # Maximum displacement from translation (voxel diagonal)
        max_trans_displacement = V * np.sqrt(3) / 2

        # Ratio tells us relative importance
        # If rot_displacement >> trans_displacement, translation matters more
        # because small translations still matter while we've "saturated" rotations
        ratio = max_trans_displacement / max_rot_displacement

        # Also consider: with many grids, we need finer discrimination
        # More grids → we're slicing SO(3) finer → rotations more important
        grid_factor = min(1.0, 60 / self.n_grids)  # diminishes above 60

        # Also: coarser grids (small G) → each voxel is larger →
        # translations within voxel have more impact
        coarseness = 32 / G  # normalized to "typical" grid size of 32

        # Combine factors
        # Base: rotation slightly more important
        base_rot = 0.6
        base_trans = 0.4

        # Adjust for coarseness (coarser → more translation weight)
        coarse_adjust = 0.15 * (coarseness - 1)  # ±0.15 adjustment

        # Adjust for grid count (more grids → more rotation weight)
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

        # Rotation distance: geodesic on SO(3), normalized to [0, 1]
        dot = np.abs(np.dot(q1, q2))
        rot_dist = 2 * np.arccos(np.clip(dot, 0, 1)) / np.pi

        # Translation distance: toroidal, normalized to [0, 1]
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
        """Generate 60 icosahedral rotation quaternions."""
        phi = (1 + np.sqrt(5)) / 2

        # 12 icosahedron vertices as rotation axes
        verts = []
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                verts.append(np.array([0, s1, s2 * phi]))
                verts.append(np.array([s1, s2 * phi, 0]))
                verts.append(np.array([s2 * phi, 0, s1]))
        verts = [v / np.linalg.norm(v) for v in verts]

        quats = [np.array([1, 0, 0, 0], dtype="f4")]  # identity

        # 5-fold rotations (72°, 144°, 216°, 288°) around each vertex
        for v in verts:
            for k in [1, 2, 3, 4]:
                angle = k * 2 * np.pi / 5
                half = angle / 2
                q = np.array([np.cos(half), *(np.sin(half) * v)], dtype="f4")
                quats.append(self._canonicalize_quat(q))

        # Deduplicate (should get exactly 60)
        unique = []
        for q in quats:
            if not any(np.abs(np.abs(np.dot(q, u)) - 1) < 1e-5 for u in unique):
                unique.append(q)

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

        # Phase 1 & 2: Icosahedral rotations, multiple offset passes
        while self._max_computed < min(n, len(icosa_quats) * 4):  # up to 240
            cycle = self._max_computed // 60  # which pass through icosahedral
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

        # Phase 3: Full SE(3) sampling for anything beyond 240
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

        # Pairwise distance statistics
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

        # Covering radius estimate (max gap)
        test_configs = [self._random_config() for _ in range(1000)]
        max_gap = max(self.min_distance_to_set(c) for c in test_configs)
        print(f"  Estimated covering radius: {max_gap:.4f}")

        return dists


def compute_transforms(self):
    """Drop-in replacement for Simulation.compute_transforms()"""
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


# Test coverage quality
if __name__ == "__main__":
    sampler = HierarchicalSE3Sampler()

    for n in [16, 32, 60, 64, 120, 128]:
        configs = sampler.compute_hierarchical(n)

        # Compute min pairwise distance (separation quality)
        min_pair = float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                d = sampler.config_distance_analytic(configs[i], configs[j])
                min_pair = min(min_pair, d)

        print(f"n={n:3d}: min_pairwise_dist={min_pair:.4f}")
