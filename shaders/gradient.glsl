#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float voxelSize;

#if DIM == 3
layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rgba32f, binding = 1) writeonly uniform image3D gradientTexture;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rgba32f, binding = 1) writeonly uniform image2D gradientTexture;
#define IVEC_TYPE ivec2
#endif

// Wrapper to handle bounds checking logic cleanly
IVEC_TYPE wrap(IVEC_TYPE pos) {
#if DIM == 3
  return (pos % tensorDimensions + tensorDimensions) % tensorDimensions;
#else
  return (pos % tensorDimensions.xy + tensorDimensions.xy) %
         tensorDimensions.xy;
#endif
}

float getVal(IVEC_TYPE pos) { return imageLoad(inputTexture, wrap(pos)).x; }

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
  // Define offsets for 3D
  IVEC_TYPE offX = ivec3(1, 0, 0);
  IVEC_TYPE offY = ivec3(0, 1, 0);
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
  // Define offsets for 2D
  IVEC_TYPE offX = ivec2(1, 0);
  IVEC_TYPE offY = ivec2(0, 1);
#endif

  // Now offX and offY are valid types for both dimensions
  float dx = (getVal(pos + offX) - getVal(pos - offX)) / (2.0 * voxelSize);
  float dy = (getVal(pos + offY) - getVal(pos - offY)) / (2.0 * voxelSize);
  float dz = 0.0;

#if DIM == 3
  IVEC_TYPE offZ = ivec3(0, 0, 1);
  dz = (getVal(pos + offZ) - getVal(pos - offZ)) / (2.0 * voxelSize);

  vec3 gradient = vec3(dx, dy, dz);
  float mag = length(gradient);
  imageStore(gradientTexture, pos, vec4(gradient, mag));
#else
  vec2 gradient = vec2(dx, dy);
  float mag = length(gradient);
  // Store 2D gradient, 0 for Z, and magnitude in Alpha
  imageStore(gradientTexture, pos, vec4(gradient, 0.0, mag));
#endif
}