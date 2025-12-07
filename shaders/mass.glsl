#version 460
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PositionBuffer {
  vec4 positions[];
};

layout(std430, binding = 1) buffer MassBuffer { uint mass_grid[]; };

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float voxelSize;

ivec2 worldToVoxel(vec2 worldPos) { return ivec2(worldPos / voxelSize); }

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= positions.length())
    return;

  vec4 particle = positions[gID];
  ivec2 voxelPos = worldToVoxel(particle.xy);

  // Wrap coordinates
  voxelPos = (voxelPos % tensorDimensions.xy + tensorDimensions.xy) %
             tensorDimensions.xy;

  int idx = voxelPos.y * tensorDimensions.x + voxelPos.x;
  // Scale mass to uint (multiply by 1000 or whatever precision you need)
  uint massInt = uint(particle.w * 1000.0);
  atomicAdd(mass_grid[idx], massInt);
}