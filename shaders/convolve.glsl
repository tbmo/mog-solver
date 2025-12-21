// Linear-Sparse-Grid Direct Convolution Shader (3D only)
// Replaces FFT -> Green's -> IFFT with direct spatial convolution
// Convolves mass density with precomputed Green's gradient kernel
// #version 460, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float G;
layout(location = 3) uniform float worldSize;

// Mass density input (from scatter step)
layout(std430, binding = 0) readonly buffer MassBuffer {
    uint mass_grid[];
};

// Precomputed Green's gradient kernel (real-space impulse response)
// Shape is (gridSize, gridSize, gridSize, 3) - stores (dG/dx, dG/dy, dG/dz)
layout(std430, binding = 1) readonly buffer GreenKernel {
    float green_kernel[];
};

// Output: force field
layout(rgba32f, binding = 0) writeonly uniform image3D forceTexture;

// Get mass at a cell, with periodic wrapping
float getMass(int gridIdx, int x, int y, int z) {
    // Periodic wrap
    x = (x % gridSize + gridSize) % gridSize;
    y = (y % gridSize + gridSize) % gridSize;
    z = (z % gridSize + gridSize) % gridSize;

    int cellsPerGrid = gridSize * gridSize * gridSize;
    int idx = gridIdx * cellsPerGrid + z * gridSize * gridSize + y * gridSize + x;
    return float(mass_grid[idx]) / 1000.0;
}

// Get Green's kernel value at offset (dx, dy, dz), component c
// Kernel is stored with periodic indexing
float getKernel(int dx, int dy, int dz, int component) {
    // Wrap to kernel coordinates (kernel is periodic)
    int kx = (dx % gridSize + gridSize) % gridSize;
    int ky = (dy % gridSize + gridSize) % gridSize;
    int kz = (dz % gridSize + gridSize) % gridSize;

    int idx = (kz * gridSize * gridSize + ky * gridSize + kx) * 3 + component;
    return green_kernel[idx];
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    // 3D: Packed layout - pos.z = gridIdx * gridSize + localZ
    int gridIdx = pos.z / gridSize;
    int localZ = pos.z % gridSize;

    if (pos.x >= gridSize || pos.y >= gridSize || gridIdx >= nGrids)
        return;

    int outX = pos.x;
    int outY = pos.y;
    int outZ = localZ;

    // Convolve: F(x) = sum over y of: mass(y) * kernel(x - y)
    vec3 force = vec3(0.0);

    for (int kz = 0; kz < gridSize; kz++) {
        for (int ky = 0; ky < gridSize; ky++) {
            for (int kx = 0; kx < gridSize; kx++) {
                float m = getMass(gridIdx, kx, ky, kz);
                if (m > 0.0) {
                    int dx = outX - kx;
                    int dy = outY - ky;
                    int dz = outZ - kz;

                    force.x += m * getKernel(dx, dy, dz, 0);
                    force.y += m * getKernel(dx, dy, dz, 1);
                    force.z += m * getKernel(dx, dy, dz, 2);
                }
            }
        }
    }

    // Apply gravitational constant
    force *= G;

    float mag = length(force);
    imageStore(forceTexture, pos, vec4(force, mag));
}
