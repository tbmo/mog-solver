"""
Force law probe v2: Use a massive central clump, measure force on test particle.

This is closer to how the simulation actually works - collective mass creates
a potential field that individual particles respond to.
"""

import numpy as np
import matplotlib.pyplot as plt


def set_uniform(prog, name, value):
    if name in prog:
        prog[name] = value


def probe_force_law_v2(
    sim, n_samples=30, r_min=0.5, r_max=5.0, n_source_particles=10000
):
    """
    Create a dense spherical clump of particles at center,
    measure force on a single test particle at various distances.

    This tests F(r) for a point mass equivalent.
    For a sphere of mass M, force outside should be F = GM/r²
    """

    # Save original state
    original_pos = np.frombuffer(sim.pos_buf.read(), dtype="f4").copy()
    original_vel = np.frombuffer(sim.vel_buf.read(), dtype="f4").copy()
    original_num = sim.num_particles

    center = sim.world_size / 2.0
    voxel = sim.voxel_size

    # Create source mass: tight sphere at center
    clump_radius = 0.5 * voxel  # Half a cell - very compact

    np.random.seed(42)

    # Uniform sphere via rejection
    source_pos = []
    while len(source_pos) < n_source_particles:
        p = (np.random.rand(3) - 0.5) * 2 * clump_radius
        if np.linalg.norm(p) < clump_radius:
            source_pos.append(p + center)
    source_pos = np.array(source_pos, dtype="f4")

    total_mass = n_source_particles * sim.cfg.get("particle_mass", 1.0)
    print(f"Source: {n_source_particles} particles, total mass = {total_mass}")
    print(f"Clump radius: {clump_radius / voxel:.2f} cells")

    separations = np.linspace(r_min, r_max, n_samples) * voxel
    forces = []

    for r in separations:
        # Create particle array: source clump + 1 test particle
        n_total = n_source_particles + 1

        pos = np.zeros((n_total, 4), dtype="f4")
        pos[:n_source_particles, :3] = source_pos
        pos[:n_source_particles, 3] = sim.cfg.get("particle_mass", 1.0)

        # Test particle at distance r along x-axis
        pos[n_source_particles] = [
            center + r,
            center,
            center,
            sim.cfg.get("particle_mass", 1.0),
        ]

        vel = np.zeros((n_total, 4), dtype="f4")

        # Resize buffers if needed (or just use existing if big enough)
        sim.pos_buf.write(pos.tobytes())
        sim.vel_buf.write(vel.tobytes())
        sim.num_particles = n_total

        # Run force computation
        gs = sim.grid_size
        ng = sim.n_grids
        dispatch = (max(1, gs // 8), max(1, gs // 8), gs * ng)

        sim.clear_mass()
        sim.pos_buf.bind_to_storage_buffer(0)
        sim.mass_buf.bind_to_storage_buffer(1)
        sim.offset_buf.bind_to_storage_buffer(2)
        sim.rotation_buf.bind_to_storage_buffer(3)
        set_uniform(sim.prog_mass, "gridSize", gs)
        set_uniform(sim.prog_mass, "nGrids", ng)
        set_uniform(sim.prog_mass, "voxelSize", sim.voxel_size)
        set_uniform(sim.prog_mass, "worldSize", sim.world_size)
        sim.prog_mass.run((n_total + 255) // 256)
        sim.ctx.memory_barrier()

        sim.mass_buf.bind_to_storage_buffer(0)
        sim.complex_tex_a.bind_to_image(1, read=False, write=True)
        set_uniform(sim.prog_complex, "gridSize", gs)
        set_uniform(sim.prog_complex, "nGrids", ng)
        sim.prog_complex.run(*dispatch)
        sim.ctx.memory_barrier()

        spectrum_tex = sim.run_fft_3d_batched(
            sim.complex_tex_a, sim.complex_tex_b, forward=True
        )

        spectrum_tex.bind_to_image(0, read=True, write=True)
        set_uniform(sim.prog_fourier, "gridSize", gs)
        set_uniform(sim.prog_fourier, "nGrids", ng)
        sim.prog_fourier.run(*dispatch)
        sim.ctx.memory_barrier()

        spectrum_tex.bind_to_image(0, read=True, write=False)
        sim.grad_comp_x.bind_to_image(1, read=False, write=True)
        sim.grad_comp_y.bind_to_image(2, read=False, write=True)
        sim.grad_comp_z.bind_to_image(3, read=False, write=True)
        set_uniform(sim.prog_greens, "gridSize", gs)
        set_uniform(sim.prog_greens, "nGrids", ng)
        set_uniform(sim.prog_greens, "G", sim.G)
        set_uniform(sim.prog_greens, "worldSize", sim.world_size)
        sim.prog_greens.run(*dispatch)
        sim.ctx.memory_barrier()

        real_x = sim.run_fft_3d_batched(
            sim.grad_comp_x, sim.complex_tex_b, forward=False
        )
        real_y = sim.run_fft_3d_batched(
            sim.grad_comp_y, sim.complex_tex_a, forward=False
        )
        real_z = sim.run_fft_3d_batched(
            sim.grad_comp_z, sim.complex_tex_b, forward=False
        )

        real_x.bind_to_image(0, read=True, write=False)
        real_y.bind_to_image(1, read=True, write=False)
        real_z.bind_to_image(2, read=True, write=False)
        sim.gradient_tex.bind_to_image(3, read=False, write=True)
        set_uniform(sim.prog_pack, "gridSize", gs)
        set_uniform(sim.prog_pack, "nGrids", ng)
        sim.prog_pack.run(*dispatch)
        sim.ctx.memory_barrier()

        # Get force on test particle via update with dt=1
        sim.pos_buf.bind_to_storage_buffer(0)
        sim.vel_buf.bind_to_storage_buffer(1)
        sim.offset_buf.bind_to_storage_buffer(2)
        sim.rotation_buf.bind_to_storage_buffer(3)
        sim.gradient_tex.bind_to_image(0, read=True, write=False)

        set_uniform(sim.prog_update, "deltaTime", 1.0)
        set_uniform(sim.prog_update, "gridSize", gs)
        set_uniform(sim.prog_update, "nGrids", ng)
        set_uniform(sim.prog_update, "worldSize", sim.world_size)
        set_uniform(sim.prog_update, "voxelSize", sim.voxel_size)
        set_uniform(sim.prog_update, "damping", 1.0)
        sim.prog_update.run((n_total + 255) // 256)
        sim.ctx.memory_barrier()

        # Read test particle's velocity (= force since dt=1, m=1)
        vel_data = np.frombuffer(sim.vel_buf.read(), dtype="f4").reshape(-1, 4)

        # Test particle is last one
        fx = -vel_data[n_source_particles, 0]
        fy = -vel_data[n_source_particles, 1]
        fz = -vel_data[n_source_particles, 2]

        # Force should be negative (toward center) so fx should be negative
        # We want magnitude of radial force
        f_radial = -fx  # Positive if pointing toward center (correct direction)
        f_mag = np.sqrt(fx**2 + fy**2 + fz**2)

        forces.append(f_radial)
        print(f"r={r / voxel:.2f} cells: F_radial={f_radial:.4f}, |F|={f_mag:.4f}")

    # Restore
    sim.pos_buf.write(original_pos.tobytes())
    sim.vel_buf.write(original_vel.tobytes())
    sim.num_particles = original_num

    separations = np.array(separations)
    forces = np.array(forces)

    # Theory: F = G * M / r^2
    r = separations
    G = sim.G
    M = total_mass
    f_theory = G * M / (r**2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(separations / voxel, forces, "bo-", ms=6, label="Measured")
    ax.plot(separations / voxel, f_theory, "r--", lw=2, label=f"GM/r² (M={M:.0f})")
    ax.set_xlabel("Distance from center (cell widths)")
    ax.set_ylabel("Radial Force")
    ax.set_title("Force vs Distance from Central Mass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mask = forces > 0
    ax.loglog(separations[mask] / voxel, forces[mask], "bo-", ms=6, label="Measured")
    ax.loglog(separations / voxel, f_theory, "r--", lw=2, label="GM/r²")
    ax.set_xlabel("Distance (cell widths)")
    ax.set_ylabel("Radial Force")
    ax.set_title("Log-Log Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fit power law to measured data
    if np.sum(mask) > 2:
        log_r = np.log(separations[mask])
        log_f = np.log(forces[mask])
        slope, intercept = np.polyfit(log_r, log_f, 1)
        print(f"\nFitted power law: F ∝ r^{slope:.2f}")
        print(f"(Should be -2 for 1/r² gravity)")

    plt.tight_layout()
    plt.savefig("force_law_v2.png", dpi=150)
    print("\nSaved force_law_v2.png")

    return separations, forces
