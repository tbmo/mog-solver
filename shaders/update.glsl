// update.glsl - NGP version
// #version 460, DIM, N_GRIDS, GRID_SIZE injected by Python

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(std430, binding = 2) readonly buffer OffsetBuffer { vec4 offsets[]; };

layout(std430, binding = 3) readonly buffer RotationBuffer {
  vec4 rotations[];
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
  int x = int(worldPos.x / voxelSize) % gridSize;
  int y = int(worldPos.y / voxelSize) % gridSize;
  return imageLoad(gradientTexture, ivec3(x, y, gridIdx)).xy;
}

#else
mat3 getRotation(int gridIdx) {
  vec4 col0 = rotations[gridIdx * 3 + 0];
  vec4 col1 = rotations[gridIdx * 3 + 1];
  vec4 col2 = rotations[gridIdx * 3 + 2];
  return mat3(col0.xyz, col1.xyz, col2.xyz);
}

layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;

vec3 sampleForce3D(vec3 worldPos, int gridIdx) {
  int x = int(worldPos.x / voxelSize) % gridSize;
  int y = int(worldPos.y / voxelSize) % gridSize;
  int z = int(worldPos.z / voxelSize) % gridSize;
  int gz = gridIdx * gridSize + z;
  return imageLoad(gradientTexture, ivec3(x, y, gz)).xyz;
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

    vec2 centered = pos - worldSize * 0.5;
    vec2 rotated = rot * centered;
    vec2 samplePos = rotated + worldSize * 0.5 - offset;
    samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);

    vec2 gridForce = sampleForce2D(samplePos, gridIdx);
    vec2 worldForce = transpose(rot) * gridForce;
    totalForce += worldForce;
  }

  vec2 avgForce = totalForce / float(nGrids);

  vel -= avgForce * deltaTime;
  pos += vel * deltaTime;
  pos = mod(pos, worldSize);

  positions[gID] = vec4(pos, 0.0, mass);
  velocities[gID] = vec4(vel, 0.0, 0.0);

#else
  vec3 pos = pData.xyz;
  vec3 vel = vData.xyz;
  float mass = pData.w;

  vec3 totalForce = vec3(0.0);

  for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
    vec3 offset = offsets[gridIdx].xyz;
    mat3 rot = getRotation(gridIdx);

    vec3 centered = pos - worldSize * 0.5;
    vec3 rotated = rot * centered;
    vec3 samplePos = rotated + worldSize * 0.5 - offset;
    samplePos = mod(mod(samplePos, worldSize) + worldSize, worldSize);

    vec3 gridForce = sampleForce3D(samplePos, gridIdx);
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