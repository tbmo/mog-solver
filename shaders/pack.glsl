#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(location = 0) uniform ivec3 tensorDimensions;

#if DIM == 3
layout(rg32f, binding = 0) readonly uniform image3D realGradX;
layout(rg32f, binding = 1) readonly uniform image3D realGradY;
layout(rg32f, binding = 2) readonly uniform image3D realGradZ;
layout(rgba32f, binding = 3) writeonly uniform image3D finalForce;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 0) readonly uniform image2D realGradX;
layout(rg32f, binding = 1) readonly uniform image2D realGradY;
layout(rgba32f, binding = 3) writeonly uniform image2D finalForce;
#define IVEC_TYPE ivec2
#endif

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

// Normalization Factor (1/N)
#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
  float N = float(tensorDimensions.x * tensorDimensions.y * tensorDimensions.z);
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
  float N = float(tensorDimensions.x * tensorDimensions.y);
#endif
  float scale = 1.0 / N;

  float fx = imageLoad(realGradX, pos).x * scale;
  float fy = imageLoad(realGradY, pos).x * scale;
  float fz = 0.0;

#if DIM == 3
  fz = imageLoad(realGradZ, pos).x * scale;
  vec3 force = vec3(fx, fy, fz);
  float mag = length(force);
  imageStore(finalForce, pos, vec4(force, mag));
#else
  vec2 force = vec2(fx, fy);
  float mag = length(force);
  imageStore(finalForce, pos, vec4(force, 0.0, mag));
#endif
}