// Linear-Sparse-Grid Fourier Manipulation Shader
// Zeros DC mode (k=0) for all grids to prevent divergence
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)

layout(rg32f, binding = 0) uniform image3D spectrumTex;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    // Bounds check
    if (any(greaterThanEqual(pos, tensorDimensions)))
        return;

    // Zero DC mode (x=0, y=0) for each grid (all Z values)
    if (pos.x == 0 && pos.y == 0) {
        imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
    }
}
