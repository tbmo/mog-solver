#version 460
// Injected: #define DIM 2 or 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;

#if DIM == 3
layout(rg32f, binding = 0) uniform image3D spectrumTex;
#define IVEC_TYPE ivec3
#else
layout(rg32f, binding = 0) uniform image2D spectrumTex;
#define IVEC_TYPE ivec2
#endif

void main() {
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
#endif

#if DIM == 3
  if (pos == ivec3(0, 0, 0)) {
    imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
  }
#else
  if (pos == ivec2(0, 0)) {
    imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
  }
#endif
}
