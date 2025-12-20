// Linear-Sparse-Grid Batched 3D FFT Shader
// Operates on packed texture (X, Y, Z * n_grids)
// Each grid's Z range is [gridIdx * gridSize, (gridIdx+1) * gridSize)
// FFT operates within each grid independently to avoid spectral leakage
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int stage;
layout(location = 2) uniform int direction;  // 1 = forward, -1 = inverse
layout(location = 3) uniform int axis;       // 0 = X, 1 = Y, 2 = Z
layout(location = 4) uniform int nGrids;

layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outputTexture;

const float PI = 3.14159265359;
const float PI2 = 2.0 * PI;

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 compute_twiddle(int k, int stageSize) {
    if (k == 0)
        return vec2(1.0, 0.0);
    if (k == stageSize / 2)
        return vec2(-1.0, 0.0);

    float phase = -float(direction) * PI2 * float(k) / float(stageSize);
    float c = cos(phase);
    float s = sin(phase);

    if (abs(c) < 1e-6) c = 0.0;
    if (abs(s) < 1e-6) s = 0.0;
    return vec2(c, s);
}

uint bit_reverse(uint x, uint n) {
    uint result = 0;
    for (uint i = 0; i < n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void main() {
    // Dispatch is (gs/8, gs/8, nGrids) - each z-slice of threads handles one grid
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int gridIdx = int(gl_GlobalInvocationID.z);

    if (gridIdx >= nGrids)
        return;
    if (pos.x >= gridSize || pos.y >= gridSize)
        return;

    uint n = uint(log2(float(gridSize)));

    // For each (x, y) thread, we process the entire Z column within this grid
    // This thread handles all Z values for position (pos.x, pos.y) in grid gridIdx

    for (int localZ = 0; localZ < gridSize; localZ++) {
        // Global position in packed texture
        ivec3 globalPos = ivec3(pos.x, pos.y, gridIdx * gridSize + localZ);

        // Get the axis index for the current axis
        int axisIndex;
        if (axis == 0) axisIndex = pos.x;
        else if (axis == 1) axisIndex = pos.y;
        else axisIndex = localZ;

        // Stage 0: Bit reversal
        if (stage == 0) {
            uint revIndex = bit_reverse(uint(axisIndex), n);
            ivec3 dstPos;
            if (axis == 0) dstPos = ivec3(int(revIndex), pos.y, gridIdx * gridSize + localZ);
            else if (axis == 1) dstPos = ivec3(pos.x, int(revIndex), gridIdx * gridSize + localZ);
            else dstPos = ivec3(pos.x, pos.y, gridIdx * gridSize + int(revIndex));

            vec2 val = imageLoad(inputTexture, globalPos).xy;
            imageStore(outputTexture, dstPos, vec4(val, 0.0, 0.0));
        } else {
            // Butterfly stages
            int stageSize = 1 << stage;
            int halfStageSize = stageSize >> 1;

            // Only process if we're handling the lower half of a butterfly pair
            if (axisIndex >= gridSize / 2)
                continue;

            int t = axisIndex;
            int i1 = (t / halfStageSize) * stageSize + (t % halfStageSize);
            int i2 = i1 + halfStageSize;
            int k = t % halfStageSize;

            if (i2 >= gridSize)
                continue;

            // Compute positions for butterfly pair
            ivec3 pos1, pos2;
            if (axis == 0) {
                pos1 = ivec3(i1, pos.y, gridIdx * gridSize + localZ);
                pos2 = ivec3(i2, pos.y, gridIdx * gridSize + localZ);
            } else if (axis == 1) {
                pos1 = ivec3(pos.x, i1, gridIdx * gridSize + localZ);
                pos2 = ivec3(pos.x, i2, gridIdx * gridSize + localZ);
            } else {
                pos1 = ivec3(pos.x, pos.y, gridIdx * gridSize + i1);
                pos2 = ivec3(pos.x, pos.y, gridIdx * gridSize + i2);
            }

            vec2 p = imageLoad(inputTexture, pos1).xy;
            vec2 q = imageLoad(inputTexture, pos2).xy;

            vec2 w = compute_twiddle(k, stageSize);
            vec2 temp = complex_mul(q, w);

            if (abs(temp.x) < 1e-6) temp.x = 0.0;
            if (abs(temp.y) < 1e-6) temp.y = 0.0;

            imageStore(outputTexture, pos1, vec4(p + temp, 0.0, 0.0));
            imageStore(outputTexture, pos2, vec4(p - temp, 0.0, 0.0));
        }
    }
}
