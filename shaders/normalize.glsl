#version 460
layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform ivec3 tensorDimensions;

layout(rg32f, binding = 0) uniform image2D fieldTexture;

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  int N = tensorDimensions.x * tensorDimensions.y;
  float scale = 1.0 / float(N);

  vec2 val = imageLoad(fieldTexture, pos).xy;
  imageStore(fieldTexture, pos, vec4(val * scale, 0.0, 0.0));
}