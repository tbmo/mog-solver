layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float G;
layout(location = 3) uniform float worldSize;
layout(location = 4) uniform float epsilon = 0.0001;

layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outGradX;
layout(rg32f, binding = 2) writeonly uniform image3D outGradY;
layout(rg32f, binding = 3) writeonly uniform image3D outGradZ;

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
  float geoFactor = -2.0 * PI; // was 4.0

  vec2 rhoHat = imageLoad(inputTexture, pos).xy;
  vec2 phiHat = vec2(0.0);

  if (k2 > epsilon) {
    phiHat = rhoHat * (geoFactor * G / k2);
  }
  // float cellSize = worldSize / float(gridSize);
  // float a = cellSize * 0.3; // tune this: 0.5–2.0 cells
  // float k2_soft = k2 + 1.0 / (a * a);
  // phiHat = rhoHat * (geoFactor * G / k2_soft);

  vec2 gradXHat = mult_ik(phiHat, kx);
  vec2 gradYHat = mult_ik(phiHat, ky);

  imageStore(outGradX, pos, vec4(gradXHat, 0.0, 0.0));
  imageStore(outGradY, pos, vec4(gradYHat, 0.0, 0.0));

  vec2 gradZHat = mult_ik(phiHat, kz);
  imageStore(outGradZ, pos, vec4(gradZHat, 0.0, 0.0));
}

// void main() {
//   ivec3 pos = ivec3(gl_GlobalInvocationID);
//   ivec3 texSize = imageSize(inputTexture);

//   if (any(greaterThanEqual(pos, texSize)))
//     return;

//   float kx = getWaveNumber(pos.x, gridSize, worldSize);
//   float ky = getWaveNumber(pos.y, gridSize, worldSize);

//   int localZ = pos.z % gridSize;
//   float kz = getWaveNumber(localZ, gridSize, worldSize);

//   float k2 = kx * kx + ky * ky + kz * kz;

//   // Calculate the physical size of a single voxel
//   float dx = worldSize / float(gridSize);

//   // 1. CHOOSE YOUR SOFTENING RADIUS 'a'
//   // 1.5 to 2.0 voxels is the mathematical "sweet spot" for particle-mesh
//   (PM)
//   // stability. Anything less than 1.0 voxel will cause sub-grid collapse.
//   float a = 0.5 * dx;

//   vec2 rhoHat = imageLoad(inputTexture, pos).xy;
//   vec2 phiHat = vec2(0.0);

//   // 2. APPLY SPECTRAL PLUMMER SOFTENING
//   // This smoothly dampens force at sub-grid scales without destroying
//   // long-range orbital forces.
//   if (k2 > epsilon) {
//     // Standard Poisson Green's function is: -4.0 * PI * G / k^2
//     // We multiply by an exponential screening factor: exp(-k^2 * a^2)
//     // This acts as a low-pass filter to prevent sub-pixel singularities.
//     float spectralFilter = exp(-k2 * a * a);

//     // NOTE: Your original geoFactor was -2.0 * PI.
//     // Standard 3D Poisson equation is -4.0 * PI. If your gravity felt weak,
//     // this is why!
//     float geoFactor = -4.0 * PI;

//     phiHat = rhoHat * ((geoFactor * G / k2) * spectralFilter);
//   }

//   // Calculate gradients (forces) from the softened potential
//   vec2 gradXHat = mult_ik(phiHat, kx);
//   vec2 gradYHat = mult_ik(phiHat, ky);

//   imageStore(outGradX, pos, vec4(gradXHat, 0.0, 0.0));
//   imageStore(outGradY, pos, vec4(gradYHat, 0.0, 0.0));

//   vec2 gradZHat = mult_ik(phiHat, kz);
//   imageStore(outGradZ, pos, vec4(gradZHat, 0.0, 0.0));
// }