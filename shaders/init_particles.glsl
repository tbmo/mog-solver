// init_particles.glsl - GPU Perlin noise particle initialization
// #version 460, DIM injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(location = 0) uniform uint numParticles;
layout(location = 1) uniform float worldSize;
layout(location = 2) uniform float noiseScale;      // Controls noise frequency
layout(location = 3) uniform float densityContrast; // Higher = more clustered
layout(location = 4) uniform uint seed;
layout(location = 5) uniform float baseMass;
layout(location = 6) uniform
    float spawnBuffer; // Fraction of worldSize to keep clear from edges
layout(location = 7) uniform uint spawnShape; // 0 = cube, 1 = sphere

// ============================================================================
// Hash functions for randomness
// ============================================================================

uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

uint hash2(uint x, uint y) {
  return hash(x ^ (hash(y) + 0x9e3779b9u + (x << 6) + (x >> 2)));
}

uint hash3(uint x, uint y, uint z) {
  return hash(hash2(x, y) ^ (hash(z) + 0x9e3779b9u));
}

float rand(uint s) { return float(hash(s)) / 4294967295.0; }

vec2 rand2(uint s) { return vec2(rand(s), rand(hash(s))); }

vec3 rand3(uint s) { return vec3(rand(s), rand(hash(s)), rand(hash(hash(s)))); }

// ============================================================================
// Perlin noise implementation
// ============================================================================

vec2 grad2(ivec2 p, uint s) {
  uint h = hash2(uint(p.x) + s, uint(p.y) + s * 7u);
  float angle = float(h) * 6.28318530718 / 4294967295.0;
  return vec2(cos(angle), sin(angle));
}

vec3 grad3(ivec3 p, uint s) {
  uint h = hash3(uint(p.x) + s, uint(p.y) + s * 7u, uint(p.z) + s * 13u);
  // Generate random unit vector on sphere
  float theta = float(h & 0xFFFFu) * 6.28318530718 / 65535.0;
  float phi = acos(2.0 * float(h >> 16) / 65535.0 - 1.0);
  return vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

float fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float perlin2D(vec2 p, uint s) {
  ivec2 i0 = ivec2(floor(p));
  vec2 f = fract(p);

  vec2 u = vec2(fade(f.x), fade(f.y));

  float n00 = dot(grad2(i0, s), f);
  float n10 = dot(grad2(i0 + ivec2(1, 0), s), f - vec2(1.0, 0.0));
  float n01 = dot(grad2(i0 + ivec2(0, 1), s), f - vec2(0.0, 1.0));
  float n11 = dot(grad2(i0 + ivec2(1, 1), s), f - vec2(1.0, 1.0));

  float nx0 = mix(n00, n10, u.x);
  float nx1 = mix(n01, n11, u.x);

  return mix(nx0, nx1, u.y);
}

float perlin3D(vec3 p, uint s) {
  ivec3 i0 = ivec3(floor(p));
  vec3 f = fract(p);

  vec3 u = vec3(fade(f.x), fade(f.y), fade(f.z));

  float n000 = dot(grad3(i0, s), f);
  float n100 = dot(grad3(i0 + ivec3(1, 0, 0), s), f - vec3(1.0, 0.0, 0.0));
  float n010 = dot(grad3(i0 + ivec3(0, 1, 0), s), f - vec3(0.0, 1.0, 0.0));
  float n110 = dot(grad3(i0 + ivec3(1, 1, 0), s), f - vec3(1.0, 1.0, 0.0));
  float n001 = dot(grad3(i0 + ivec3(0, 0, 1), s), f - vec3(0.0, 0.0, 1.0));
  float n101 = dot(grad3(i0 + ivec3(1, 0, 1), s), f - vec3(1.0, 0.0, 1.0));
  float n011 = dot(grad3(i0 + ivec3(0, 1, 1), s), f - vec3(0.0, 1.0, 1.0));
  float n111 = dot(grad3(i0 + ivec3(1, 1, 1), s), f - vec3(1.0, 1.0, 1.0));

  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);

  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);

  return mix(nxy0, nxy1, u.z);
}

// Fractal Brownian Motion for more natural clustering
float fbm2D(vec2 p, uint s, int octaves) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;

  for (int i = 0; i < octaves; i++) {
    value += amplitude * perlin2D(p * frequency, s + uint(i) * 1000u);
    amplitude *= 0.5;
    frequency *= 2.0;
  }

  return value;
}

float fbm3D(vec3 p, uint s, int octaves) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;

  for (int i = 0; i < octaves; i++) {
    value += amplitude * perlin3D(p * frequency, s + uint(i) * 1000u);
    amplitude *= 0.5;
    frequency *= 2.0;
  }

  return value;
}

// ============================================================================
// Rejection sampling with Perlin noise density field
// ============================================================================

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= numParticles)
    return;

  uint particleSeed = seed + gID * 12345u;

  // Compute spawn region: shrink from edges by spawnBuffer fraction
  float minBound = spawnBuffer * worldSize;
  float maxBound = (1.0 - spawnBuffer) * worldSize;
  float spawnRange = maxBound - minBound;

#if DIM == 2
  // Rejection sampling: generate candidates until one passes density test
  vec2 pos;
  uint attempts = 0u;
  const uint maxAttempts = 64u;

  // Center of the spawn region
  vec2 center = vec2(worldSize * 0.5);
  float radius = spawnRange * 0.5;

  while (attempts < maxAttempts) {
    vec2 candidate;

    if (spawnShape == 1u) {
      // Sphere/circle mode: rejection sample within unit disk, then scale
      vec2 diskPoint;
      uint diskAttempts = 0u;
      do {
        // Generate point in [-1, 1]^2
        diskPoint = rand2(particleSeed + attempts * 1000u + diskAttempts * 100u) * 2.0 - 1.0;
        diskAttempts++;
      } while (dot(diskPoint, diskPoint) > 1.0 && diskAttempts < 16u);

      // Scale to spawn region centered in world
      candidate = center + diskPoint * radius;
    } else {
      // Cube mode: uniform in spawn region
      candidate = rand2(particleSeed + attempts * 1000u) * spawnRange + minBound;
    }

    // Sample density field using FBM
    vec2 noiseCoord = candidate * noiseScale / worldSize;
    float density = fbm2D(noiseCoord, seed, 4);

    // Map [-0.5, 0.5] range to [0, 1] and apply contrast
    density = (density + 0.5);
    density = pow(clamp(density, 0.0, 1.0), densityContrast);

    // Rejection test
    float threshold = rand(particleSeed + attempts * 1000u + 500u);
    if (threshold < density) {
      pos = candidate;
      break;
    }

    attempts++;
  }

  // Fallback if max attempts reached (shouldn't happen often)
  if (attempts >= maxAttempts) {
    if (spawnShape == 1u) {
      // Fallback for sphere: use sqrt for uniform disk distribution
      float r = sqrt(rand(particleSeed)) * radius;
      float theta = rand(hash(particleSeed)) * 6.28318530718;
      pos = center + vec2(cos(theta), sin(theta)) * r;
    } else {
      pos = rand2(particleSeed) * spawnRange + minBound;
    }
  }

  // Small random velocity for initial dynamics
  vec2 vel = (rand2(particleSeed + 999u) - 0.5) * 1.0 * worldSize;

  positions[gID] = vec4(pos, 0.0, baseMass);
  velocities[gID] = vec4(vel, 0.0, 0.0);

#else
  // 3D version
  vec3 pos;
  uint attempts = 0u;
  const uint maxAttempts = 64u;

  // Center of the spawn region
  vec3 center = vec3(worldSize * 0.5);
  float radius = spawnRange * 0.5;

  while (attempts < maxAttempts) {
    vec3 candidate;

    if (spawnShape == 1u) {
      // Sphere mode: rejection sample within unit sphere, then scale
      vec3 spherePoint;
      uint sphereAttempts = 0u;
      do {
        // Generate point in [-1, 1]^3
        spherePoint = rand3(particleSeed + attempts * 1000u + sphereAttempts * 100u) * 2.0 - 1.0;
        sphereAttempts++;
      } while (dot(spherePoint, spherePoint) > 1.0 && sphereAttempts < 16u);

      // Scale to spawn region centered in world
      candidate = center + spherePoint * radius;
    } else {
      // Cube mode: uniform in spawn region
      candidate = rand3(particleSeed + attempts * 1000u) * spawnRange + minBound;
    }

    vec3 noiseCoord = candidate * noiseScale / worldSize;
    float density = fbm3D(noiseCoord, seed, 4);

    density = (density + 0.5);
    density = pow(clamp(density, 0.0, 1.0), densityContrast);

    float threshold = rand(particleSeed + attempts * 1000u + 500u);
    if (threshold < density) {
      pos = candidate;
      break;
    }

    attempts++;
  }

  if (attempts >= maxAttempts) {
    if (spawnShape == 1u) {
      // Fallback for sphere: use cube root for uniform sphere distribution
      float r = pow(rand(particleSeed), 1.0 / 3.0) * radius;
      float theta = rand(hash(particleSeed)) * 6.28318530718;
      float phi = acos(2.0 * rand(hash(hash(particleSeed))) - 1.0);
      pos = center + vec3(
          sin(phi) * cos(theta),
          sin(phi) * sin(theta),
          cos(phi)
      ) * r;
    } else {
      pos = rand3(particleSeed) * spawnRange + minBound;
    }
  }

  vec3 vel = (rand3(particleSeed + 999u) - 0.5) * 1.0 * worldSize;

  positions[gID] = vec4(pos, baseMass);
  velocities[gID] = vec4(vel, 0.0);
#endif
}
