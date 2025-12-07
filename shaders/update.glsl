#version 460
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };

layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(location = 0) uniform float deltaTime;
layout(location = 1) uniform ivec3 tensorDimensions;
layout(location = 2) uniform float worldSize;
layout(location = 3) uniform float voxelSize;

layout(rgba32f, binding = 0) readonly uniform image2D gradientTexture;

ivec2 worldToVoxel(vec2 worldPos) { return ivec2(worldPos / voxelSize); }

ivec2 wrapCoords(ivec2 pos) {
  return (pos % tensorDimensions.xy + tensorDimensions.xy) %
         tensorDimensions.xy;
}

vec2 getGradientAtPosition(vec2 worldPos) {
  ivec2 voxelPos = worldToVoxel(worldPos);
  voxelPos = wrapCoords(voxelPos);
  return imageLoad(gradientTexture, voxelPos).xy;
}

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= positions.length())
    return;

  vec4 particle = positions[gID];
  vec2 pos = particle.xy;
  float mass = particle.w;

  vec2 gradient = getGradientAtPosition(pos);
  vec4 velocity = velocities[gID];

  // F = -grad(phi), a = F/m but for gravity a = -grad(phi) directly
  velocity.xy -= gradient * deltaTime;

  pos += velocity.xy * deltaTime;

  // Wrap around boundaries
  pos = mod(pos, worldSize);
  if (pos.x < 0)
    pos.x += worldSize;
  if (pos.y < 0)
    pos.y += worldSize;

  positions[gID] = vec4(pos, 0.0, mass);
  velocities[gID] = velocity;
}