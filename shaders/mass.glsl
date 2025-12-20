// Linear-Sparse-Grid Mass Scatter Shader
// Scatters particle mass to ALL grids with diagonal offsets
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

ivec2 worldToVoxel(vec2 worldPos) {
    return ivec2(worldPos / voxelSize);
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= positions.length())
        return;

    vec4 particle = positions[gID];
    vec2 basePos = particle.xy;
    uint massInt = uint(particle.w * 1.0);

    int gridSizeX = tensorDimensions.x;
    int gridSizeY = tensorDimensions.y;
    int nGrids = tensorDimensions.z;

    // Scatter to ALL grids with their respective offsets
    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        // Apply offset: local_pos = particle.pos - offset[gridIdx]
        vec2 localPos = basePos - offsets[gridIdx];

        // Periodic boundary wrapping
        localPos = mod(localPos, worldSize);
        if (localPos.x < 0.0) localPos.x += worldSize;
        if (localPos.y < 0.0) localPos.y += worldSize;

        // Convert to voxel coordinates
        ivec2 voxelPos = worldToVoxel(localPos);

        // Wrap voxel coordinates
        voxelPos = (voxelPos % ivec2(gridSizeX, gridSizeY) + ivec2(gridSizeX, gridSizeY)) % ivec2(gridSizeX, gridSizeY);

        // Flatten index: gridIdx * (gridSizeX * gridSizeY) + y * gridSizeX + x
        int idx = gridIdx * (gridSizeX * gridSizeY) + voxelPos.y * gridSizeX + voxelPos.x;

        // Atomic add mass
        atomicAdd(mass_grid[idx], massInt);
    }
}
