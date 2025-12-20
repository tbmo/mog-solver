// Linear-Sparse-Grid Particle Update Shader
// Gathers forces from ALL grids with BILINEAR (2D) or TRILINEAR (3D) interpolation
// Averages forces and applies leapfrog integration
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(std430, binding = 2) readonly buffer OffsetBuffer {
    vec4 offsets[];  // [N_GRIDS] diagonal offsets (vec4 for alignment)
};

layout(location = 0) uniform float deltaTime;
layout(location = 1) uniform int gridSize;
layout(location = 2) uniform int nGrids;
layout(location = 3) uniform float worldSize;
layout(location = 4) uniform float voxelSize;

#if DIM == 2
// 2D: Single 3D texture with Z as batch dimension
layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;

vec2 sampleForce2D(vec2 worldPos, int gridIdx) {
    vec2 voxelCoord = worldPos / voxelSize;
    vec2 voxelFloor = floor(voxelCoord);
    vec2 frac = voxelCoord - voxelFloor;

    int x0 = int(mod(voxelFloor.x, float(gridSize)));
    int y0 = int(mod(voxelFloor.y, float(gridSize)));
    int x1 = int(mod(voxelFloor.x + 1.0, float(gridSize)));
    int y1 = int(mod(voxelFloor.y + 1.0, float(gridSize)));
    if (x0 < 0) x0 += gridSize;
    if (y0 < 0) y0 += gridSize;
    if (x1 < 0) x1 += gridSize;
    if (y1 < 0) y1 += gridSize;

    vec2 f00 = imageLoad(gradientTexture, ivec3(x0, y0, gridIdx)).xy;
    vec2 f10 = imageLoad(gradientTexture, ivec3(x1, y0, gridIdx)).xy;
    vec2 f01 = imageLoad(gradientTexture, ivec3(x0, y1, gridIdx)).xy;
    vec2 f11 = imageLoad(gradientTexture, ivec3(x1, y1, gridIdx)).xy;

    vec2 f0 = mix(f00, f10, frac.x);
    vec2 f1 = mix(f01, f11, frac.x);
    return mix(f0, f1, frac.y);
}

#else
// 3D: Multiple 3D textures, one per grid (bound to consecutive image units)
layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture0;
layout(rgba32f, binding = 1) readonly uniform image3D gradientTexture1;
layout(rgba32f, binding = 2) readonly uniform image3D gradientTexture2;
layout(rgba32f, binding = 3) readonly uniform image3D gradientTexture3;
layout(rgba32f, binding = 4) readonly uniform image3D gradientTexture4;
layout(rgba32f, binding = 5) readonly uniform image3D gradientTexture5;
layout(rgba32f, binding = 6) readonly uniform image3D gradientTexture6;
layout(rgba32f, binding = 7) readonly uniform image3D gradientTexture7;

vec3 sampleFromTexture(int texIdx, ivec3 p) {
    // Manual dispatch to correct texture
    if (texIdx == 0) return imageLoad(gradientTexture0, p).xyz;
    if (texIdx == 1) return imageLoad(gradientTexture1, p).xyz;
    if (texIdx == 2) return imageLoad(gradientTexture2, p).xyz;
    if (texIdx == 3) return imageLoad(gradientTexture3, p).xyz;
    if (texIdx == 4) return imageLoad(gradientTexture4, p).xyz;
    if (texIdx == 5) return imageLoad(gradientTexture5, p).xyz;
    if (texIdx == 6) return imageLoad(gradientTexture6, p).xyz;
    return imageLoad(gradientTexture7, p).xyz;
}

vec3 sampleForce3D(vec3 worldPos, int gridIdx) {
    vec3 voxelCoord = worldPos / voxelSize;
    vec3 voxelFloor = floor(voxelCoord);
    vec3 frac = voxelCoord - voxelFloor;

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

    // Sample 8 corners
    vec3 f000 = sampleFromTexture(gridIdx, ivec3(x0, y0, z0));
    vec3 f100 = sampleFromTexture(gridIdx, ivec3(x1, y0, z0));
    vec3 f010 = sampleFromTexture(gridIdx, ivec3(x0, y1, z0));
    vec3 f110 = sampleFromTexture(gridIdx, ivec3(x1, y1, z0));
    vec3 f001 = sampleFromTexture(gridIdx, ivec3(x0, y0, z1));
    vec3 f101 = sampleFromTexture(gridIdx, ivec3(x1, y0, z1));
    vec3 f011 = sampleFromTexture(gridIdx, ivec3(x0, y1, z1));
    vec3 f111 = sampleFromTexture(gridIdx, ivec3(x1, y1, z1));

    // Trilinear interpolation
    vec3 f00 = mix(f000, f100, frac.x);
    vec3 f10 = mix(f010, f110, frac.x);
    vec3 f01 = mix(f001, f101, frac.x);
    vec3 f11 = mix(f011, f111, frac.x);

    vec3 f0 = mix(f00, f10, frac.y);
    vec3 f1 = mix(f01, f11, frac.y);

    return mix(f0, f1, frac.z);
}
#endif

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= positions.length())
        return;

    vec4 pData = positions[gID];
    vec4 vData = velocities[gID];

#if DIM == 2
    vec2 pos = pData.xy;
    vec2 vel = vData.xy;
    float mass = pData.w;

    vec2 totalForce = vec2(0.0);

    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        vec2 offset = offsets[gridIdx].xy;
        vec2 samplePos = pos - offset;
        samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);
        totalForce += sampleForce2D(samplePos, gridIdx);
    }

    vec2 avgForce = totalForce / float(nGrids);

    vel -= avgForce * deltaTime;
    pos += vel * deltaTime;
    pos = mod(pos, worldSize);

    positions[gID] = vec4(pos, 0.0, mass);
    velocities[gID] = vec4(vel, 0.0, 0.0);

#else  // DIM == 3
    vec3 pos = pData.xyz;
    vec3 vel = vData.xyz;
    float mass = pData.w;

    vec3 totalForce = vec3(0.0);

    for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
        vec3 offset = offsets[gridIdx].xyz;
        vec3 samplePos = pos - offset;
        samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);
        totalForce += sampleForce3D(samplePos, gridIdx);
    }

    vec3 avgForce = totalForce / float(nGrids);

    vel -= avgForce * deltaTime;
    pos += vel * deltaTime;
    pos = mod(pos, worldSize);

    positions[gID] = vec4(pos, mass);
    velocities[gID] = vec4(vel, 0.0);
#endif
}
