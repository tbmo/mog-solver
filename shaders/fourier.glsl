// Linear-Sparse-Grid Fourier Manipulation Shader
// Zeros DC mode (k=0) for all grids to prevent divergence
// For 2D: Z is batch dimension
// For 3D: Single grid at a time
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;

layout(rg32f, binding = 0) uniform image3D spectrumTex;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 texSize = imageSize(spectrumTex);

    if (any(greaterThanEqual(pos, texSize)))
        return;

#if DIM == 2
    // 2D: Zero DC at (0,0) for each grid slice
    if (pos.x == 0 && pos.y == 0) {
        imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
    }
#else
    // 3D: Zero DC at (0,0,0)
    if (pos.x == 0 && pos.y == 0 && pos.z == 0) {
        imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
    }
#endif
}
