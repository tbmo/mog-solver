// Linear-Sparse-Grid Direct Convolution Shader (3D only)
// Replaces FFT -> Green's -> IFFT with direct spatial convolution
// Convolves mass density with precomputed Green's gradient kernel
// #version 460, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float G;
layout(location = 3) uniform float worldSize;

// Mass density input (from scatter step)
layout(std430, binding = 0) readonly buffer MassBuffer {
    uint mass_grid[];
};

// Precomputed Green's gradient kernel (real-space impulse response)
// Shape is (gridSize, gridSize, gridSize, 4) - stores (dG/dx, dG/dy, dG/dz, 0) as vec4
layout(std430, binding = 1) readonly buffer GreenKernel {
    vec4 green_kernel[];
};

// Output: force field
layout(rgba32f, binding = 0) writeonly uniform image3D forceTexture;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    int gridIdx = pos.z / gridSize;
    int localZ = pos.z % gridSize;

    if (pos.x >= gridSize || pos.y >= gridSize || gridIdx >= nGrids)
        return;

    int outX = pos.x;
    int outY = pos.y;
    int outZ = localZ;

    int cellsPerGrid = gridSize * gridSize * gridSize;
    int massBase = gridIdx * cellsPerGrid;

    vec3 force = vec3(0.0);

    // Convolve: F(x) = sum over y of: mass(y) * kernel(x - y)
    for (int kz = 0; kz < gridSize; kz++) {
        int dz = outZ - kz;
        int wz = ((dz % gridSize) + gridSize) % gridSize;
        int kernelZBase = wz * gridSize * gridSize;
        int massZBase = kz * gridSize * gridSize;

        for (int ky = 0; ky < gridSize; ky++) {
            int dy = outY - ky;
            int wy = ((dy % gridSize) + gridSize) % gridSize;
            int kernelYBase = kernelZBase + wy * gridSize;
            int massYBase = massZBase + ky * gridSize;

            for (int kx = 0; kx < gridSize; kx++) {
                int massIdx = massBase + massYBase + kx;
                float m = float(mass_grid[massIdx]) * 0.001;

                if (m > 0.0) {
                    int dx = outX - kx;
                    int wx = ((dx % gridSize) + gridSize) % gridSize;
                    int kernelIdx = kernelYBase + wx;

                    force += m * green_kernel[kernelIdx].xyz;
                }
            }
        }
    }

    force *= G;
    float mag = length(force);
    imageStore(forceTexture, pos, vec4(force, mag));
}
