#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PositionBuffer { vec4 positions[]; };
layout(std430, binding = 1) buffer VelocityBuffer { vec4 velocities[]; };

layout(location = 0) uniform float deltaTime;
layout(location = 1) uniform ivec3 tensorDimensions;
layout(location = 2) uniform float worldSize;
layout(location = 3) uniform float voxelSize;

#if DIM == 3
layout(rgba32f, binding = 0) readonly uniform image3D gradientTexture;
#define IVEC_TYPE ivec3
#define VEC_TYPE vec3
#else
layout(rgba32f, binding = 0) readonly uniform image2D gradientTexture;
#define IVEC_TYPE ivec2
#define VEC_TYPE vec2
#endif

IVEC_TYPE worldToVoxel(VEC_TYPE worldPos) {
  return IVEC_TYPE(worldPos / voxelSize);
}

void main() {
  uint gID = gl_GlobalInvocationID.x;
  if (gID >= positions.length())
    return;

  vec4 pData = positions[gID];
  vec4 vData = velocities[gID];

// Extract Logic
#if DIM == 3
  VEC_TYPE pos = pData.xyz;
  VEC_TYPE vel = vData.xyz;
  float mass = pData.w;
#else
  VEC_TYPE pos = pData.xy;
  VEC_TYPE vel = vData.xy;
  float mass = pData.w; // In 2D we stored mass in W
#endif

  // Get Gradient
  IVEC_TYPE voxelPos = worldToVoxel(pos);

#if DIM == 3
  voxelPos =
      (voxelPos % tensorDimensions + tensorDimensions) % tensorDimensions;
  vec3 grad = imageLoad(gradientTexture, voxelPos).xyz;

  vel -= grad * deltaTime;
  pos += vel * deltaTime;

  // 3D Wrap
  pos = mod(pos, worldSize);

  // Write Back
  positions[gID] = vec4(pos, mass);
  velocities[gID] = vec4(vel, 0.0);

#else
  voxelPos = (voxelPos % tensorDimensions.xy + tensorDimensions.xy) %
             tensorDimensions.xy;
  vec2 grad = imageLoad(gradientTexture, voxelPos).xy;

  vel -= grad * deltaTime;
  pos += vel * deltaTime;

  // 2D Wrap
  pos = mod(pos, worldSize);

  // Write Back
  positions[gID] = vec4(pos, 0.0, mass);
  velocities[gID] = vec4(vel, 0.0, 0.0);
#endif
}