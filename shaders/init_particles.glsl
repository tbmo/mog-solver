// init_particles.glsl - Sphere initialization
// #version 460, DIM injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(location = 0) uniform uint numParticles;
layout(location = 1) uniform float worldSize;
layout(location = 2) uniform
    float noiseScale; // Used as sphere radius (fraction of worldSize)
layout(location = 3) uniform
    float densityContrast; // Unused but kept for compatibility
layout(location = 4) uniform uint seed;
layout(location = 5) uniform float baseMass;
layout(location = 6) uniform
    float spawnBuffer; // Unused but kept for compatibility

uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

float rand(uint s) { return float(hash(s)) / 4294967295.0; }

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= numParticles)
    return;

  uint particleSeed = seed + gID * 12345u;

  float sphereRadius = noiseScale * worldSize * 0.5; // noiseScale as fraction
  vec3 center = vec3(worldSize * 0.5);

#if DIM == 2
  // 2D: uniform disk
  float r = sphereRadius * sqrt(rand(particleSeed)); // sqrt for uniform area
  float theta = rand(hash(particleSeed)) * 6.28318530718;

  vec2 pos = center.xy + r * vec2(cos(theta), sin(theta));
  vec2 vel = vec2(0.0);

  positions[gID] = vec4(pos, 0.0, baseMass);
  velocities[gID] = vec4(vel, 0.0, 0.0);

#else
  // 3D: uniform sphere (rejection sampling or analytic)
  // Using analytic method for uniform volume distribution

  float u = rand(particleSeed);
  float v = rand(hash(particleSeed));
  float w = rand(hash(hash(particleSeed)));

  // Uniform in volume: r proportional to cbrt(uniform)
  float r = sphereRadius * pow(u, 1.0 / 3.0);

  // Uniform on sphere surface
  float cosTheta = 2.0 * v - 1.0;
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  float phi = 2.0 * 3.14159265359 * w;

  vec3 pos =
      center + r * vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

  // Zero initial velocity (cold start) or small random
  vec3 vel = vec3(0.0);

  positions[gID] = vec4(pos, baseMass);
  velocities[gID] = vec4(vel, 0.0);
#endif
}