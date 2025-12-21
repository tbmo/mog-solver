// Linear-Sparse-Grid Green's Function Shader
// Applies Green's function and spectral differentiation in Fourier space
// For 2D: Z is batch dimension
// For 3D: Packed layout (X, Y, Z*nGrids) - each grid's local Z for wave numbers
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float G;
layout(location = 3) uniform float worldSize;
layout(location = 4) uniform float epsilon = 1e-3;

layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outGradX;
layout(rg32f, binding = 2) writeonly uniform image3D outGradY;
#if DIM == 3
layout(rg32f, binding = 3) writeonly uniform image3D outGradZ;
#endif

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;

float getWaveNumber(int i, int N, float L) {
  int halfN = N / 2;
  int shifted = (i <= halfN) ? i : (i - N);
  return float(shifted) * (TWO_PI / L);
}

vec2 mult_ik(vec2 c, float k) { return vec2(-c.y * k, c.x * k); }

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 texSize = imageSize(inputTexture);

  if (any(greaterThanEqual(pos, texSize)))
    return;

  float kx = getWaveNumber(pos.x, gridSize, worldSize);
  float ky = getWaveNumber(pos.y, gridSize, worldSize);

  // 3D: pos.z encodes gridIdx * gridSize + localZ
  // We need localZ for the wave number calculation
  int localZ = pos.z % gridSize;
  float kz = getWaveNumber(localZ, gridSize, worldSize);
  float k2 = kx * kx + ky * ky + kz * kz;
  float geoFactor = -4.0 * PI;

  vec2 rhoHat = imageLoad(inputTexture, pos).xy;
  vec2 phiHat = vec2(0.0);

  if (k2 > epsilon) {
    phiHat = rhoHat * (geoFactor * G / k2);
  }

  vec2 gradXHat = mult_ik(phiHat, kx);
  vec2 gradYHat = mult_ik(phiHat, ky);

  imageStore(outGradX, pos, vec4(gradXHat, 0.0, 0.0));
  imageStore(outGradY, pos, vec4(gradYHat, 0.0, 0.0));

  vec2 gradZHat = mult_ik(phiHat, kz);
  imageStore(outGradZ, pos, vec4(gradZHat, 0.0, 0.0));
}
