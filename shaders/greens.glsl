#version 460
layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float G;
layout(location = 2) uniform float worldSize;

layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image2D outputTexture;

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;

float getWaveNumber(int i, int N, float L) {
  int halfN = N / 2;
  int shifted = (i <= halfN) ? i : (i - N);
  return float(shifted) * (TWO_PI / L);
}

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  int Nx = tensorDimensions.x;
  int Ny = tensorDimensions.y;
  float L = worldSize;

  vec2 rhoFreq = imageLoad(inputTexture, pos).xy;

  float kx = getWaveNumber(pos.x, Nx, L);
  float ky = getWaveNumber(pos.y, Ny, L);
  float k2 = kx * kx + ky * ky;

  vec2 phiFreq = vec2(0.0);

  // 2D Green's function: -2πG/k² (not 4π like 3D)
  if (k2 > 1e-9) {
    float factor = -2.0 * PI * G / k2;
    phiFreq = rhoFreq * factor;
  }

  imageStore(outputTexture, pos, vec4(phiFreq, 0.0, 0.0));
}