#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(std430, binding = 0) readonly buffer MassBuffer { uint mass_grid[]; };

#if DIM == 3
layout(rg32f, binding = 1) writeonly uniform image3D complexTexture;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 1) writeonly uniform image2D complexTexture;
#define IVEC_TYPE ivec2
#endif

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
  // Flatten 3D index for reading from buffer
  int idx = pos.z * (tensorDimensions.x * tensorDimensions.y) +
            pos.y * tensorDimensions.x + pos.x;
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
  int idx = pos.y * tensorDimensions.x + pos.x;
#endif

  float mass = float(mass_grid[idx]) / 1000.0;
  imageStore(complexTexture, pos, vec4(mass, 0.0, 0.0, 0.0));
}