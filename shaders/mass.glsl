layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PositionBuffer {
  vec4 positions[];
};

layout(std430, binding = 1) buffer MassBuffer { uint mass_grid[]; };

layout(std430, binding = 2) readonly buffer OffsetBuffer { vec4 offsets[]; };

layout(std430, binding = 3) readonly buffer RotationBuffer {
  vec4 rotations[];
};

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;
layout(location = 2) uniform float voxelSize;
layout(location = 3) uniform float worldSize;

mat3 getRotation(int gridIdx) {
  vec4 col0 = rotations[gridIdx * 3 + 0];
  vec4 col1 = rotations[gridIdx * 3 + 1];
  vec4 col2 = rotations[gridIdx * 3 + 2];
  return mat3(col0.xyz, col1.xyz, col2.xyz);
}

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= positions.length())
    return;

  vec4 particle = positions[gID];
  float mass = particle.w;

  vec3 basePos = particle.xyz;
  int cellsPerGrid = gridSize * gridSize * gridSize;

  for (int gridIdx = 0; gridIdx < nGrids; gridIdx++) {
    vec3 offset = offsets[gridIdx].xyz;
    mat3 rot = getRotation(gridIdx);

    vec3 centered = basePos - worldSize * 0.5;
    vec3 rotated = rot * centered;
    vec3 localPos = rotated + worldSize * 0.5 - offset;

    localPos = mod(mod(localPos, worldSize) + worldSize, worldSize);

    int x = int(localPos.x / voxelSize) % gridSize;
    int y = int(localPos.y / voxelSize) % gridSize;
    int z = int(localPos.z / voxelSize) % gridSize;

    atomicAdd(mass_grid[gridIdx * cellsPerGrid + z * gridSize * gridSize +
                        y * gridSize + x],
              uint(mass * 1000.0));
  }
}