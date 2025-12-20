// Linear-Sparse-Grid Particle Update Shader
// Gathers forces from ALL grids with BILINEAR interpolation
// Averages forces and applies leapfrog integration
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(std430, binding = 2) readonly buffer OffsetBuffer {
    vec2 offsets[];  // [N_GRIDS] diagonal offsets in world space
};

layout(location = 0) uniform float deltaTime;
layout(location = 1) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)
layout(location = 2) uniform float worldSize;
layout(location = 3) uniform float voxelSize;

layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;

// Bilinear interpolation helper for 2D sampling within a specific grid slice
vec2 sampleForce(vec2 worldPos, int gridIdx) {
    int gridSizeX = tensorDimensions.x;
    int gridSizeY = tensorDimensions.y;

    // Convert to continuous voxel coordinates
    vec2 voxelCoord = worldPos / voxelSize;

    // Get integer and fractional parts
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

    // Sample four corners from the specific grid slice (z = gridIdx)
    vec2 f00 = imageLoad(gradientTexture, ivec3(x0, y0, gridIdx)).xy;
    vec2 f10 = imageLoad(gradientTexture, ivec3(x1, y0, gridIdx)).xy;
    vec2 f01 = imageLoad(gradientTexture, ivec3(x0, y1, gridIdx)).xy;
    vec2 f11 = imageLoad(gradientTexture, ivec3(x1, y1, gridIdx)).xy;

    // Bilinear interpolation
    vec2 f0 = mix(f00, f10, frac.x);
    vec2 f1 = mix(f01, f11, frac.x);
    return mix(f0, f1, frac.y);
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= positions.length())
        return;

    vec4 pData = positions[gID];
    vec4 vData = velocities[gID];

    vec2 pos = pData.xy;
    vec2 vel = vData.xy;
    float mass = pData.w;

    int nGrids = tensorDimensions.z;

    // Gather forces from ALL grids and average
    vec2 totalForce = vec2(0.0);

    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        // Calculate sample position with offset: sample_pos = pos - offset[gridIdx]
        vec2 samplePos = pos - offsets[gridIdx];

        // Periodic wrapping (handle negative values properly)
        samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);

        // Sample with bilinear interpolation
        vec2 gridForce = sampleForce(samplePos, gridIdx);
        totalForce += gridForce;
    }

    // Average force across all grids
    vec2 avgForce = totalForce / float(nGrids);

    // Leapfrog integration
    vel -= avgForce * deltaTime;
    pos += vel * deltaTime;

    // Periodic boundary wrapping
    pos = mod(pos, worldSize);

    // Write back
    positions[gID] = vec4(pos, 0.0, mass);
    velocities[gID] = vec4(vel, 0.0, 0.0);
}
