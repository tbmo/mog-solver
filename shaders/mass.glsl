// Linear-Sparse-Grid Mass Scatter Shader
// Scatters particle mass to ALL grids with diagonal offsets
// Uses BILINEAR (2D) or TRILINEAR (3D) CIC mass assignment
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PositionBuffer {
    vec4 positions[];
};

layout(std430, binding = 1) buffer MassBuffer {
    uint mass_grid[];  // Flattened [N_GRIDS, GRID_SIZE^DIM]
};

layout(std430, binding = 2) readonly buffer OffsetBuffer {
    vec4 offsets[];  // [N_GRIDS] diagonal offsets in world space (vec4 for alignment)
};

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float voxelSize;
layout(location = 3) uniform float worldSize;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= positions.length())
        return;

    vec4 particle = positions[gID];
    float mass = particle.w;

#if DIM == 2
    vec2 basePos = particle.xy;
    int cellsPerGrid = gridSize * gridSize;

    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        vec2 offset = offsets[gridIdx].xy;
        vec2 localPos = basePos - offset;

        // Periodic wrapping
        localPos = mod(mod(localPos, worldSize) + worldSize, worldSize);

        // Continuous voxel coordinates
        vec2 voxelCoord = localPos / voxelSize;
        vec2 voxelFloor = floor(voxelCoord);
        vec2 frac = voxelCoord - voxelFloor;

        // Four corners with periodic wrap
        int x0 = int(mod(voxelFloor.x, float(gridSize)));
        int y0 = int(mod(voxelFloor.y, float(gridSize)));
        int x1 = int(mod(voxelFloor.x + 1.0, float(gridSize)));
        int y1 = int(mod(voxelFloor.y + 1.0, float(gridSize)));
        if (x0 < 0) x0 += gridSize;
        if (y0 < 0) y0 += gridSize;
        if (x1 < 0) x1 += gridSize;
        if (y1 < 0) y1 += gridSize;

        // Bilinear weights
        float w00 = (1.0 - frac.x) * (1.0 - frac.y);
        float w10 = frac.x * (1.0 - frac.y);
        float w01 = (1.0 - frac.x) * frac.y;
        float w11 = frac.x * frac.y;

        int baseIdx = gridIdx * cellsPerGrid;
        atomicAdd(mass_grid[baseIdx + y0 * gridSize + x0], uint(mass * w00));
        atomicAdd(mass_grid[baseIdx + y0 * gridSize + x1], uint(mass * w10));
        atomicAdd(mass_grid[baseIdx + y1 * gridSize + x0], uint(mass * w01));
        atomicAdd(mass_grid[baseIdx + y1 * gridSize + x1], uint(mass * w11));
    }

#else  // DIM == 3
    vec3 basePos = particle.xyz;
    int cellsPerGrid = gridSize * gridSize * gridSize;

    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        vec3 offset = offsets[gridIdx].xyz;
        vec3 localPos = basePos - offset;

        // Periodic wrapping
        localPos = mod(mod(localPos, worldSize) + worldSize, worldSize);

        // Continuous voxel coordinates
        vec3 voxelCoord = localPos / voxelSize;
        vec3 voxelFloor = floor(voxelCoord);
        vec3 frac = voxelCoord - voxelFloor;

        // Eight corners with periodic wrap
        int x0 = int(mod(voxelFloor.x, float(gridSize)));
        int y0 = int(mod(voxelFloor.y, float(gridSize)));
        int z0 = int(mod(voxelFloor.z, float(gridSize)));
        int x1 = int(mod(voxelFloor.x + 1.0, float(gridSize)));
        int y1 = int(mod(voxelFloor.y + 1.0, float(gridSize)));
        int z1 = int(mod(voxelFloor.z + 1.0, float(gridSize)));
        if (x0 < 0) x0 += gridSize;
        if (y0 < 0) y0 += gridSize;
        if (z0 < 0) z0 += gridSize;
        if (x1 < 0) x1 += gridSize;
        if (y1 < 0) y1 += gridSize;
        if (z1 < 0) z1 += gridSize;

        // Trilinear weights
        float w000 = (1.0 - frac.x) * (1.0 - frac.y) * (1.0 - frac.z);
        float w100 = frac.x * (1.0 - frac.y) * (1.0 - frac.z);
        float w010 = (1.0 - frac.x) * frac.y * (1.0 - frac.z);
        float w110 = frac.x * frac.y * (1.0 - frac.z);
        float w001 = (1.0 - frac.x) * (1.0 - frac.y) * frac.z;
        float w101 = frac.x * (1.0 - frac.y) * frac.z;
        float w011 = (1.0 - frac.x) * frac.y * frac.z;
        float w111 = frac.x * frac.y * frac.z;

        int baseIdx = gridIdx * cellsPerGrid;
        int gs2 = gridSize * gridSize;

        atomicAdd(mass_grid[baseIdx + z0 * gs2 + y0 * gridSize + x0], uint(mass * w000));
        atomicAdd(mass_grid[baseIdx + z0 * gs2 + y0 * gridSize + x1], uint(mass * w100));
        atomicAdd(mass_grid[baseIdx + z0 * gs2 + y1 * gridSize + x0], uint(mass * w010));
        atomicAdd(mass_grid[baseIdx + z0 * gs2 + y1 * gridSize + x1], uint(mass * w110));
        atomicAdd(mass_grid[baseIdx + z1 * gs2 + y0 * gridSize + x0], uint(mass * w001));
        atomicAdd(mass_grid[baseIdx + z1 * gs2 + y0 * gridSize + x1], uint(mass * w101));
        atomicAdd(mass_grid[baseIdx + z1 * gs2 + y1 * gridSize + x0], uint(mass * w011));
        atomicAdd(mass_grid[baseIdx + z1 * gs2 + y1 * gridSize + x1], uint(mass * w111));
    }
#endif
}
