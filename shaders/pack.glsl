// Linear-Sparse-Grid Force Packing Shader
// Packs real-space force components into final force texture
// Applies 1/N normalization for inverse FFT
// For 2D: Z is batch dimension
// For 3D: Single grid at a time
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;

layout(rg32f, binding = 0) readonly uniform image3D realGradX;
layout(rg32f, binding = 1) readonly uniform image3D realGradY;
#if DIM == 3
layout(rg32f, binding = 2) readonly uniform image3D realGradZ;
#endif
layout(rgba32f, binding = 3) writeonly uniform image3D finalForce;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 texSize = imageSize(realGradX);

    if (any(greaterThanEqual(pos, texSize)))
        return;

#if DIM == 2
    float N = float(gridSize * gridSize);
#else
    float N = float(gridSize * gridSize * gridSize);
#endif
    float scale = 1.0 / N;

    float fx = imageLoad(realGradX, pos).x * scale;
    float fy = imageLoad(realGradY, pos).x * scale;

#if DIM == 2
    vec2 force = vec2(fx, fy);
    float mag = length(force);
    imageStore(finalForce, pos, vec4(force, 0.0, mag));
#else
    float fz = imageLoad(realGradZ, pos).x * scale;
    vec3 force = vec3(fx, fy, fz);
    float mag = length(force);
    imageStore(finalForce, pos, vec4(force, mag));
#endif
}
