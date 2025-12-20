// update.glsl - full replacement
// Linear-Sparse-Grid Particle Update Shader
// Gathers forces from ALL grids with BILINEAR (2D) or TRILINEAR (3D)
// interpolation Applies inverse rotation to force vectors #version 460, DIM,
// N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(std430, binding = 2) readonly buffer OffsetBuffer { vec4 offsets[]; };

layout(std430, binding = 3) readonly buffer RotationBuffer {
  vec4 rotations[]; // 3 vec4s per grid
};

layout(location = 0) uniform float deltaTime;
layout(location = 1) uniform int gridSize;
layout(location = 2) uniform int nGrids;
layout(location = 3) uniform float worldSize;
layout(location = 4) uniform float voxelSize;

#if DIM == 2
mat2 getRotation(int gridIdx) {
  vec4 col0 = rotations[gridIdx * 3 + 0];
  vec4 col1 = rotations[gridIdx * 3 + 1];
  return mat2(col0.xy, col1.xy);
}

layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;

vec2 sampleForce2D(vec2 worldPos, int gridIdx) {
  vec2 voxelCoord = worldPos / voxelSize;
  vec2 voxelFloor = floor(voxelCoord);
  vec2 frac = voxelCoord - voxelFloor;

  int x0 = int(mod(voxelFloor.x, float(gridSize)));
  int y0 = int(mod(voxelFloor.y, float(gridSize)));
  int x1 = int(mod(voxelFloor.x + 1.0, float(gridSize)));
  int y1 = int(mod(voxelFloor.y + 1.0, float(gridSize)));
  if (x0 < 0)
    x0 += gridSize;
  if (y0 < 0)
    y0 += gridSize;
  if (x1 < 0)
    x1 += gridSize;
  if (y1 < 0)
    y1 += gridSize;

  vec2 f00 = imageLoad(gradientTexture, ivec3(x0, y0, gridIdx)).xy;
  vec2 f10 = imageLoad(gradientTexture, ivec3(x1, y0, gridIdx)).xy;
  vec2 f01 = imageLoad(gradientTexture, ivec3(x0, y1, gridIdx)).xy;
  vec2 f11 = imageLoad(gradientTexture, ivec3(x1, y1, gridIdx)).xy;

  vec2 f0 = mix(f00, f10, frac.x);
  vec2 f1 = mix(f01, f11, frac.x);
  return mix(f0, f1, frac.y);
}

#else
mat3 getRotation(int gridIdx) {
  vec4 col0 = rotations[gridIdx * 3 + 0];
  vec4 col1 = rotations[gridIdx * 3 + 1];
  vec4 col2 = rotations[gridIdx * 3 + 2];
  return mat3(col0.xyz, col1.xyz, col2.xyz);
}

// 3D: Single packed texture (X, Y, Z*nGrids)
layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;

vec3 sampleForce3D(vec3 worldPos, int gridIdx) {
  vec3 voxelCoord = worldPos / voxelSize;
  vec3 voxelFloor = floor(voxelCoord);
  vec3 frac = voxelCoord - voxelFloor;

  int x0 = int(mod(voxelFloor.x, float(gridSize)));
  int y0 = int(mod(voxelFloor.y, float(gridSize)));
  int z0 = int(mod(voxelFloor.z, float(gridSize)));
  int x1 = int(mod(voxelFloor.x + 1.0, float(gridSize)));
  int y1 = int(mod(voxelFloor.y + 1.0, float(gridSize)));
  int z1 = int(mod(voxelFloor.z + 1.0, float(gridSize)));
  if (x0 < 0)
    x0 += gridSize;
  if (y0 < 0)
    y0 += gridSize;
  if (z0 < 0)
    z0 += gridSize;
  if (x1 < 0)
    x1 += gridSize;
  if (y1 < 0)
    y1 += gridSize;
  if (z1 < 0)
    z1 += gridSize;

  // Convert local z to global z in packed texture
  int zBase = gridIdx * gridSize;
  int gz0 = zBase + z0;
  int gz1 = zBase + z1;

  vec3 f000 = imageLoad(gradientTexture, ivec3(x0, y0, gz0)).xyz;
  vec3 f100 = imageLoad(gradientTexture, ivec3(x1, y0, gz0)).xyz;
  vec3 f010 = imageLoad(gradientTexture, ivec3(x0, y1, gz0)).xyz;
  vec3 f110 = imageLoad(gradientTexture, ivec3(x1, y1, gz0)).xyz;
  vec3 f001 = imageLoad(gradientTexture, ivec3(x0, y0, gz1)).xyz;
  vec3 f101 = imageLoad(gradientTexture, ivec3(x1, y0, gz1)).xyz;
  vec3 f011 = imageLoad(gradientTexture, ivec3(x0, y1, gz1)).xyz;
  vec3 f111 = imageLoad(gradientTexture, ivec3(x1, y1, gz1)).xyz;

  vec3 f00 = mix(f000, f100, frac.x);
  vec3 f10 = mix(f010, f110, frac.x);
  vec3 f01 = mix(f001, f101, frac.x);
  vec3 f11 = mix(f011, f111, frac.x);

  vec3 f0 = mix(f00, f10, frac.y);
  vec3 f1 = mix(f01, f11, frac.y);

  return mix(f0, f1, frac.z);
}
#endif

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= positions.length())
    return;

  vec4 pData = positions[gID];
  vec4 vData = velocities[gID];

#if DIM == 2
  vec2 pos = pData.xy;
  vec2 vel = vData.xy;
  float mass = pData.w;

  vec2 totalForce = vec2(0.0);

  for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
    vec2 offset = offsets[gridIdx].xy;
    mat2 rot = getRotation(gridIdx);

    // Transform position to grid space (same as scatter)
    vec2 centered = pos - worldSize * 0.5;
    vec2 rotated = rot * centered;
    vec2 samplePos = rotated + worldSize * 0.5 - offset;
    samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);

    // Sample force in grid space
    vec2 gridForce = sampleForce2D(samplePos, gridIdx);

    // Rotate force back to world space (transpose = inverse for orthogonal)
    vec2 worldForce = transpose(rot) * gridForce;
    totalForce += worldForce;
  }

  vec2 avgForce = totalForce / float(nGrids);

  vel -= avgForce * deltaTime;
  pos += vel * deltaTime;
  pos = mod(pos, worldSize);

  positions[gID] = vec4(pos, 0.0, mass);
  velocities[gID] = vec4(vel, 0.0, 0.0);

#else // DIM == 3
  vec3 pos = pData.xyz;
  vec3 vel = vData.xyz;
  float mass = pData.w;

  vec3 totalForce = vec3(0.0);

  for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
    vec3 offset = offsets[gridIdx].xyz;
    mat3 rot = getRotation(gridIdx);

    // Transform position to grid space (same as scatter)
    vec3 centered = pos - worldSize * 0.5;
    vec3 rotated = rot * centered;
    vec3 samplePos = rotated + worldSize * 0.5 - offset;
    samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);

    // Sample force in grid space
    vec3 gridForce = sampleForce3D(samplePos, gridIdx);

    // Rotate force back to world space (transpose = inverse for orthogonal)
    vec3 worldForce = transpose(rot) * gridForce;
    totalForce += worldForce;
  }

  vec3 avgForce = totalForce / float(nGrids);

  vel -= avgForce * deltaTime;
  pos += vel * deltaTime;
  pos = mod(pos, worldSize);

  positions[gID] = vec4(pos, mass);
  velocities[gID] = vec4(vel, 0.0);
#endif
}