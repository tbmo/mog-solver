// Linear-Sparse-Grid Force Packing Shader
// Packs real-space force components into final force texture
// Applies 1/N normalization for inverse FFT
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)

layout(rg32f, binding = 0) readonly uniform image3D realGradX;
layout(rg32f, binding = 1) readonly uniform image3D realGradY;
layout(rgba32f, binding = 3) writeonly uniform image3D finalForce;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    // Bounds check
    if (any(greaterThanEqual(pos, tensorDimensions)))
        return;

    // Normalization factor: 1/N where N = grid_size * grid_size
    float N = float(tensorDimensions.x * tensorDimensions.y);
    float scale = 1.0 / N;

    // Extract real parts and scale
    float fx = imageLoad(realGradX, pos).x * scale;
    float fy = imageLoad(realGradY, pos).x * scale;

    vec2 force = vec2(fx, fy);
    float mag = length(force);

    imageStore(finalForce, pos, vec4(force, 0.0, mag));
}
