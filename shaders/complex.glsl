// Linear-Sparse-Grid Complex Conversion Shader
// Converts mass buffer to complex texture
// For 2D: Z dimension is batch (grid index)
// For 3D: Single grid at a time (currentGrid uniform)
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer MassBuffer {
    uint mass_grid[];
};

layout(rg32f, binding = 1) writeonly uniform image3D complexTexture;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform int currentGrid;  // Used in 3D mode

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

#if DIM == 2
    // 2D: pos.z is grid index
    if (pos.x >= gridSize || pos.y >= gridSize || pos.z >= nGrids)
        return;

    int cellsPerGrid = gridSize * gridSize;
    int idx = pos.z * cellsPerGrid + pos.y * gridSize + pos.x;

#else  // DIM == 3
    // 3D: Single grid texture, currentGrid uniform specifies which
    if (pos.x >= gridSize || pos.y >= gridSize || pos.z >= gridSize)
        return;
    if (currentGrid >= nGrids)  // Bounds check (also prevents nGrids from being optimized out)
        return;

    int cellsPerGrid = gridSize * gridSize * gridSize;
    int idx = currentGrid * cellsPerGrid + pos.z * gridSize * gridSize + pos.y * gridSize + pos.x;
#endif

    float mass = float(mass_grid[idx]) / 1000.0;
    imageStore(complexTexture, pos, vec4(mass, 0.0, 0.0, 0.0));
}
