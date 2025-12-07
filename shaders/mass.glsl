// #version and #define DIM injected by Python
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PositionBuffer {
  vec4 positions[];
};
layout(std430, binding = 1) buffer MassBuffer { uint mass_grid[]; };

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float voxelSize;

// Generic coordinate wrapper
#if DIM == 3
#define IVEC_TYPE ivec3
#define VEC_TYPE vec3
#else
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

  vec4 particle = positions[gID];

// Extract relevant dimensions
#if DIM == 3
  VEC_TYPE p = particle.xyz;
#else
  VEC_TYPE p = particle.xy;
#endif

  IVEC_TYPE voxelPos = worldToVoxel(p);

// Wrap
#if DIM == 3
  voxelPos = (voxelPos % tensorDimensions.xyz + tensorDimensions.xyz) %
             tensorDimensions.xyz;
  // Flatten 3D index: z * (w*h) + y * w + x
  int idx = voxelPos.z * (tensorDimensions.x * tensorDimensions.y) +
            voxelPos.y * tensorDimensions.x + voxelPos.x;
#else
  voxelPos = (voxelPos % tensorDimensions.xy + tensorDimensions.xy) %
             tensorDimensions.xy;
  int idx = voxelPos.y * tensorDimensions.x + voxelPos.x;
#endif

  uint massInt = uint(particle.w * 1000.0);
  atomicAdd(mass_grid[idx], massInt);
}