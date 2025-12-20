// Linear-Sparse-Grid Green's Function Shader
// Applies Green's function and spectral differentiation in Fourier space
// Broadcasts single kernel across all grids (Z = batch dimension)
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;  // (GRID_SIZE, GRID_SIZE, N_GRIDS)
layout(location = 1) uniform float G;
layout(location = 2) uniform float worldSize;
layout(location = 3) uniform float epsilon = 1e-9;

layout(rg32f, binding = 0) readonly uniform image3D inputTexture;   // Density spectrum
layout(rg32f, binding = 1) writeonly uniform image3D outGradX;      // Force X spectrum
layout(rg32f, binding = 2) writeonly uniform image3D outGradY;      // Force Y spectrum

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;

float getWaveNumber(int i, int N, float L) {
    int halfN = N / 2;
    int shifted = (i <= halfN) ? i : (i - N);
    return float(shifted) * (TWO_PI / L);
}

// Complex multiply by i*k: (x+iy) * ik = -yk + ixk
vec2 mult_ik(vec2 c, float k) {
    return vec2(-c.y * k, c.x * k);
}

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    // Bounds check
    if (any(greaterThanEqual(pos, tensorDimensions)))
        return;

    // Wave numbers computed from X,Y position only (same for all grids)
    float kx = getWaveNumber(pos.x, tensorDimensions.x, worldSize);
    float ky = getWaveNumber(pos.y, tensorDimensions.y, worldSize);
    float k2 = kx * kx + ky * ky;

    // 2D geometric factor
    float geoFactor = -2.0 * PI;

    // Read density spectrum for this grid
    vec2 rhoHat = imageLoad(inputTexture, pos).xy;
    vec2 phiHat = vec2(0.0);

    // Poisson Solve: Phi = -G * Rho / k^2
    if (k2 > epsilon) {
        phiHat = rhoHat * (geoFactor * G / k2);
    }

    // Spectral Gradient: F_x = i * kx * Phi, F_y = i * ky * Phi
    vec2 gradXHat = mult_ik(phiHat, kx);
    vec2 gradYHat = mult_ik(phiHat, ky);

    imageStore(outGradX, pos, vec4(gradXHat, 0.0, 0.0));
    imageStore(outGradY, pos, vec4(gradYHat, 0.0, 0.0));
}
