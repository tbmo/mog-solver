#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float G;
layout(location = 2) uniform float worldSize;

#if DIM == 3
layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outputTexture;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image2D outputTexture;
#define IVEC_TYPE ivec2
#endif

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;

float getWaveNumber(int i, int N, float L) {
  int halfN = N / 2;
  int shifted = (i <= halfN) ? i : (i - N);
  return float(shifted) * (TWO_PI / L);
}

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

  float k2 = 0.0;
  float geoFactor = 0.0;

#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;

  float kx = getWaveNumber(pos.x, tensorDimensions.x, worldSize);
  float ky = getWaveNumber(pos.y, tensorDimensions.y, worldSize);
  float kz = getWaveNumber(pos.z, tensorDimensions.z, worldSize);
  k2 = kx * kx + ky * ky + kz * kz;

  // 3D Green's Function: -4πG / k²
  geoFactor = -4.0 * PI;
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  float kx = getWaveNumber(pos.x, tensorDimensions.x, worldSize);
  float ky = getWaveNumber(pos.y, tensorDimensions.y, worldSize);
  k2 = kx * kx + ky * ky;

  // 2D Green's Function: -2πG / k²
  geoFactor = -2.0 * PI;
#endif

  vec2 rhoFreq = imageLoad(inputTexture, pos).xy;
  vec2 phiFreq = vec2(0.0);

  if (k2 > 1e-9) {
    float factor = geoFactor * G / k2;
    phiFreq = rhoFreq * factor;
  }

  imageStore(outputTexture, pos, vec4(phiFreq, 0.0, 0.0));
}