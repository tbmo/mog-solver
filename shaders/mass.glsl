// Linear-Sparse-Grid Mass Scatter Shader
// Scatters particle mass to ALL grids with diagonal offsets
// Uses BILINEAR (CIC) mass assignment to match bilinear gather
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PositionBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) buffer MassBuffer {
    uint mass_grid[];  // Flattened [N_GRIDS, GRID_SIZE, GRID_SIZE]
};

layout(std430, binding = 2) readonly buffer OffsetBuffer {
    vec2 offsets[];  // [N_GRIDS] diagonal offsets in world space
};

layout(location = 0) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)
layout(location = 1) uniform float voxelSize;
layout(location = 2) uniform float worldSize;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= positions.length())
        return;

    vec4 particle = positions[gID];
    vec2 basePos = particle.xy;
    float mass = particle.w;

    int gridSizeX = tensorDimensions.x;
    int gridSizeY = tensorDimensions.y;
    int nGrids = tensorDimensions.z;
    int cellsPerGrid = gridSizeX * gridSizeY;

    // Scatter to ALL grids with their respective offsets using BILINEAR (CIC) assignment
    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        // Apply offset: local_pos = particle.pos - offset[gridIdx]
        vec2 localPos = basePos - offsets[gridIdx];

        // Periodic boundary wrapping (handle negative values properly)
        localPos = mod(mod(localPos, worldSize) + worldSize, worldSize);

        // Convert to continuous voxel coordinates
        vec2 voxelCoord = localPos / voxelSize;

        // Get integer and fractional parts for bilinear weighting
        vec2 voxelFloor = floor(voxelCoord);
        vec2 frac = voxelCoord - voxelFloor;

        // Four corner indices with periodic wrapping
        int x0 = int(mod(voxelFloor.x, float(gridSizeX)));
        int y0 = int(mod(voxelFloor.y, float(gridSizeY)));
        int x1 = int(mod(voxelFloor.x + 1.0, float(gridSizeX)));
        int y1 = int(mod(voxelFloor.y + 1.0, float(gridSizeY)));

        // Handle negative modulo
        if (x0 < 0) x0 += gridSizeX;
        if (y0 < 0) y0 += gridSizeY;
        if (x1 < 0) x1 += gridSizeX;
        if (y1 < 0) y1 += gridSizeY;

        // Bilinear weights (CIC)
        float w00 = (1.0 - frac.x) * (1.0 - frac.y);
        float w10 = frac.x * (1.0 - frac.y);
        float w01 = (1.0 - frac.x) * frac.y;
        float w11 = frac.x * frac.y;

        // Flatten indices and scatter mass to 4 cells
        int baseIdx = gridIdx * cellsPerGrid;

        atomicAdd(mass_grid[baseIdx + y0 * gridSizeX + x0], uint(mass * w00));
        atomicAdd(mass_grid[baseIdx + y0 * gridSizeX + x1], uint(mass * w10));
        atomicAdd(mass_grid[baseIdx + y1 * gridSizeX + x0], uint(mass * w01));
        atomicAdd(mass_grid[baseIdx + y1 * gridSizeX + x1], uint(mass * w11));
    }
}
