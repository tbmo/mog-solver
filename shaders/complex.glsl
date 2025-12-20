// Linear-Sparse-Grid Complex Conversion Shader
// Converts batched mass buffer to 3D complex texture
// Z-dimension is grid index (batch dimension)
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)

layout(std430, binding = 0) readonly buffer MassBuffer {
    uint mass_grid[];  // Flattened [N_GRIDS, GRID_SIZE, GRID_SIZE]
};

layout(rg32f, binding = 1) writeonly uniform image3D complexTexture;

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    // Bounds check
    if (any(greaterThanEqual(pos, tensorDimensions)))
        return;

    int gridSizeX = tensorDimensions.x;
    int gridSizeY = tensorDimensions.y;

    // Flatten index: z * (gridSizeX * gridSizeY) + y * gridSizeX + x
    int idx = pos.z * (gridSizeX * gridSizeY) + pos.y * gridSizeX + pos.x;

    // Convert uint mass to float, store as real part of complex
    float mass = float(mass_grid[idx]) / 1000.0;
    imageStore(complexTexture, pos, vec4(mass, 0.0, 0.0, 0.0));
}
