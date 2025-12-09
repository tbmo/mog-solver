#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float G;
layout(location = 2) uniform float worldSize;
layout(location = 3) uniform float epsilon = 1e-6;

#if DIM == 3
layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outGradX;
layout(rg32f, binding = 2) writeonly uniform image3D outGradY;
layout(rg32f, binding = 3) writeonly uniform image3D outGradZ;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image2D outGradX;
layout(rg32f, binding = 2) writeonly uniform image2D outGradY;
#define IVEC_TYPE ivec2
#endif

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;

float getWaveNumber(int i, int N, float L) {
  int halfN = N / 2;
  int shifted = (i <= halfN) ? i : (i - N);
  return float(shifted) * (TWO_PI / L);
}

// Complex multiply by i*k: (x+iy) * ik = -yk + ixk
vec2 mult_ik(vec2 c, float k) { return vec2(-c.y * k, c.x * k); }

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

  float kx, ky, kz, k2;
  float geoFactor;

#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
  kx = getWaveNumber(pos.x, tensorDimensions.x, worldSize);
  ky = getWaveNumber(pos.y, tensorDimensions.y, worldSize);
  kz = getWaveNumber(pos.z, tensorDimensions.z, worldSize);
  k2 = kx * kx + ky * ky + kz * kz;
  geoFactor = -4.0 * PI;
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
  kx = getWaveNumber(pos.x, tensorDimensions.x, worldSize);
  ky = getWaveNumber(pos.y, tensorDimensions.y, worldSize);
  k2 = kx * kx + ky * ky;
  geoFactor = -2.0 * PI;
#endif

  vec2 rhoHat = imageLoad(inputTexture, pos).xy;
  vec2 phiHat = vec2(0.0);

  // Poisson Solve: Phi = -G * Rho / k^2
  if (k2 > epsilon) {
    phiHat = 0.9 * rhoHat * (geoFactor * G / k2);
  }

  // Spectral Gradient: F_x = i * kx * Phi
  vec2 gradXHat = mult_ik(phiHat, kx);
  vec2 gradYHat = mult_ik(phiHat, ky);

  imageStore(outGradX, pos, vec4(gradXHat, 0.0, 0.0));
  imageStore(outGradY, pos, vec4(gradYHat, 0.0, 0.0));

#if DIM == 3
  vec2 gradZHat = mult_ik(phiHat, kz);
  imageStore(outGradZ, pos, vec4(gradZHat, 0.0, 0.0));
#endif
}